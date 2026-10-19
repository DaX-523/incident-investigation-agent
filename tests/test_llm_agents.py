import pytest
from langchain_core.messages import ToolMessage

from investigator.agents.llm_agents import (
    LLMGatherer,
    LLMRefusal,
    ReasoningOutput,
    StructuredOutputError,
    _Caller,
    llm_workers,
    render_tool_result,
)
from investigator.llm.config import PRESETS, load_model_config
from investigator.llm.factory import build_chat_model
from investigator.llm.metering import UsageLedger
from investigator.llm.pricing import cost_usd
from investigator.orchestrator.context import WorkerContext
from investigator.report import CorrectiveActionOut, ReportNarrative
from investigator.runner import investigate
from investigator.scenario import load_scenario
from investigator.schemas import (
    CriticVerdict,
    HypothesisAssessment,
    HypothesisSpec,
    Task,
    TaskSpec,
    ToolCallRecord,
    TriagePlan,
)
from investigator.tools.harness import RetrySettings, ToolHarness
from investigator.tools.toolbox import ToolBox

from tests.llm_fakes import FakeModel, ai, last_call_ids

NO_SLEEP = RetrySettings(max_retries=1, timeout_s=2.0, backoff_base_s=0.0, backoff_cap_s=0.0)


# --- pricing / config / ledger ------------------------------------------------------------


def test_cost_math_with_cache_and_long_prompt_tier():
    # 10k input of which 4k cache reads, 1k output on Sonnet 5.5: 6k*2 + 4k*0.2 + 1k*10 per 1M
    assert cost_usd("claude-sonnet-5-5", 10_000, 1_000, cache_read=4_000) == pytest.approx((12_000 + 800 + 10_000) / 1e6)
    # Haiku long-prompt tier above 100k input tokens
    assert cost_usd("claude-haiku-5-5", 200_000, 0) == pytest.approx(200_000 * 0.50 / 1e6)
    assert cost_usd("claude-haiku-5-5", 50_000, 0) == pytest.approx(50_000 * 0.10 / 1e6)
    assert cost_usd("some-unknown-model", 1000, 1000) is None


TIER = {"claude-opus-5-5": 3, "claude-sonnet-5-5": 2, "claude-haiku-5-5": 1}


def test_budget_preset_is_one_tier_below_balanced():
    for role, spec in PRESETS["budget"].items():
        expected = max(1, TIER[PRESETS["balanced"][role].model] - 1)
        assert TIER[spec.model] == expected, role


def test_default_preset_is_budget_and_env_overrides_apply():
    config = load_model_config(env={})
    assert config["reasoning"].model == "claude-sonnet-5-5"
    config = load_model_config(env={"INVESTIGATOR_PRESET": "balanced", "MODEL_CRITIC": "openai:gpt-5.4", "EFFORT_CRITIC": "high"})
    assert config["critic"].provider == "openai" and config["critic"].model == "gpt-5.4"
    assert config["reasoning"].model == "claude-opus-5-5"
    with pytest.raises(ValueError):
        load_model_config("nope", env={})


def test_ledger_groups_by_role_and_prices_on_the_answering_model():
    ledger = UsageLedger()
    ledger.record("reasoning", ai(model="claude-sonnet-5-5", input_tokens=1000, output_tokens=100), "claude-sonnet-5-5")
    # Server-side fallback answered with a different model: priced on that model.
    ledger.record("critic", ai(model="claude-opus-5-5", input_tokens=1000, output_tokens=100), "claude-sonnet-5-5")
    ledger.record("critic", ai(model="mystery-model"), "mystery-model")
    by_model = ledger.by_model()
    assert set(by_model) == {"claude-sonnet-5-5", "claude-opus-5-5", "mystery-model"}
    assert by_model["claude-opus-5-5"].cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
    assert ledger.by_role()["critic"].llm_calls == 2 and ledger.by_role()["critic"].unpriced_calls == 1
    assert ledger.total().cost_usd == pytest.approx(sum(u.cost_usd for u in by_model.values()))


# --- gatherer tool loop ------------------------------------------------------------------------


def gatherer(script, scenario="inc-001", allowance=5):
    ledger = UsageLedger()
    spec = load_model_config(env={})["observability"]
    model = FakeModel(script=script)
    g = LLMGatherer("observability", _Caller("observability", spec, model, ledger))
    harness = ToolHarness(load_scenario(scenario), settings=NO_SLEEP)
    toolbox = ToolBox(harness, allowance)
    task = Task(task_id="T1", agent="observability", goal="check errors", hypothesis_ids=["H1"])
    ctx = WorkerContext(incident={"title": "x"}, tool_call_allowance=allowance)
    return g, task, ctx, toolbox, ledger, model


def submit(call_ids, summary="checkout 5xx at 22% from 14:05", stance="supports"):
    return {"name": "submit_findings", "args": {"evidence": [{"summary": summary, "source_call_ids": call_ids,
                                                             "links": [{"hypothesis_id": "H1", "stance": stance}]}]}}


def test_gatherer_runs_tools_and_cites_real_call_ids():
    def script(messages):
        ids = last_call_ids(messages)
        if not ids:
            return ai(tool_calls=[{"name": "query_metrics", "args": {"service": "checkout-svc"}},
                                  {"name": "search_logs", "args": {"service": "checkout-svc", "level": "ERROR"}}])
        return ai(tool_calls=[submit(ids)])

    g, task, ctx, toolbox, ledger, model = gatherer(script)
    outcome = g(task, ctx, toolbox)
    assert outcome.status == "ok"
    assert outcome.evidence[0].source_call_ids == ["call-0001", "call-0002"]
    assert outcome.usage.llm_calls == 2 and outcome.usage.cost_usd > 0
    assert ledger.by_role()["observability"].llm_calls == 2
    assert {t["name"] for t in model.bound_tools} >= {"query_metrics", "submit_findings"}
    assert "get_changes" not in {t["name"] for t in model.bound_tools}  # scoped toolset


def test_gatherer_rejects_tools_outside_its_scope_and_bad_submissions():
    turns = iter([
        ai(tool_calls=[{"name": "get_changes", "args": {}}]),                           # not in scope
        ai(tool_calls=[{"name": "submit_findings", "args": {"evidence": "nope"}}]),     # invalid
        ai(tool_calls=[{"name": "get_alerts", "args": {}}]),
    ])

    def script(messages):
        nxt = next(turns, None)
        return nxt if nxt is not None else ai(tool_calls=[submit(last_call_ids(messages))])

    g, task, ctx, toolbox, ledger, model = gatherer(script)
    outcome = g(task, ctx, toolbox)
    errors = [m for m in model.calls[-1] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errors) == 2
    assert outcome.evidence[0].source_call_ids == ["call-0001"]
    assert len(toolbox.records()) == 1  # the out-of-scope call never reached the harness


def test_gatherer_without_findings_raises_for_worker_retry():
    g, task, ctx, toolbox, *_ = gatherer(lambda m: ai(content="I think it's the database."))
    with pytest.raises(StructuredOutputError):
        g(task, ctx, toolbox)


def test_refusal_raises():
    g, task, ctx, toolbox, *_ = gatherer(lambda m: ai(stop_reason="refusal"))
    with pytest.raises(LLMRefusal):
        g(task, ctx, toolbox)


def test_tool_allowance_is_respected_by_the_loop():
    def script(messages):
        if len(last_call_ids(messages)) < 10 and not any("allowance used up" in str(m.content) for m in messages):
            return ai(tool_calls=[{"name": "get_alerts", "args": {}}])
        return ai(tool_calls=[submit(last_call_ids(messages)[:1])])

    g, task, ctx, toolbox, *_ = gatherer(script, allowance=3)
    outcome = g(task, ctx, toolbox)
    assert len(toolbox.records()) == 3 and outcome.evidence


def test_gatherer_brief_lists_queries_the_team_already_ran():
    g, task, ctx, *_ = gatherer(lambda m: ai())
    ctx = ctx.model_copy(update={"tool_calls": [
        ToolCallRecord(call_id="c1", tool="get_changes", args={"service": "api"}, ok=True, empty=True),
        ToolCallRecord(call_id="c2", tool="search_logs", ok=False, error_kind="timeout"),
    ]})
    brief = g._brief(task, ctx)
    assert 'get_changes {"service": "api"} -> no results' in brief
    assert "search_logs {} -> failed (timeout)" in brief


def test_prompt_caching_only_on_tool_loops(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = load_model_config(env={})
    assert build_chat_model(config["baseline"], cache=True).model_kwargs["cache_control"] == {"type": "ephemeral"}
    workers = llm_workers(config, UsageLedger())
    assert "cache_control" in workers.gatherers["observability"].caller.model.model_kwargs
    assert "cache_control" not in workers.reasoner.caller.model.model_kwargs


def test_large_tool_results_are_truncated_with_a_visible_marker():
    harness = ToolHarness(load_scenario("inc-013"), settings=NO_SLEEP)
    result = harness.call("query_metrics", {"service": "dialer"})
    text = render_tool_result(result, max_chars=500)
    assert len(text) < 700 and "investigator truncated" in text


# --- full graph with fake LLMs ------------------------------------------------------------------


def fake_models():
    def planner(schema, messages):
        return TriagePlan(summary="checkout failing", affected_services=["checkout-svc"],
                          hypotheses=[HypothesisSpec(key="deploy", statement="Recent checkout deploy broke checkout"),
                                      HypothesisSpec(key="psp", statement="Payment provider outage")],
                          tasks=[TaskSpec(key="o", agent="observability", goal="errors", hypothesis_keys=["deploy", "psp"]),
                                 TaskSpec(key="c", agent="change_history", goal="deploys", hypothesis_keys=["deploy"])])

    def gather_script(messages):
        ids = last_call_ids(messages)
        return ai(tool_calls=[submit(ids)]) if ids else ai(tool_calls=[{"name": "get_alerts", "args": {}}])

    def topo_script(messages):
        ids = last_call_ids(messages)
        return ai(tool_calls=[submit(ids, "checkout-svc depends on payments-svc", "neutral")]) if ids \
            else ai(tool_calls=[{"name": "get_topology", "args": {}}])

    def change_script(messages):
        ids = last_call_ids(messages)
        return ai(tool_calls=[submit(ids, "checkout-svc v2.14.0 deployed 14:01")]) if ids \
            else ai(tool_calls=[{"name": "get_changes", "args": {}}])

    def reasoner(schema, messages):
        text = messages[-1].content
        evidence_ids = [w.strip(":") for w in text.split() if w.startswith("E") and w.strip(":")[1:].isdigit()]
        return ReasoningOutput(assessments=[
            HypothesisAssessment(hypothesis_id="H1", statement="deploy", rank=1, status="supported", confidence="high",
                                 supporting=evidence_ids[:2], rationale="errors begin right after deploy"),
            HypothesisAssessment(hypothesis_id="H2", statement="psp", rank=2, status="refuted", confidence="medium",
                                 contradicting=evidence_ids[:1], rationale="payments healthy"),
        ])

    def critic(schema, messages):
        return CriticVerdict(verdict="accept")

    def reporter(schema, messages):
        return ReportNarrative(summary="Deploy broke checkout.", root_cause="checkout-svc v2.14.0",
                               corrective_actions=[CorrectiveActionOut(rank=1, category="mitigate", action="Roll back",
                                                                       rationale="restores service", evidence_ids=["E1", "E999"])])

    return {
        "triage": FakeModel(structured=planner),
        "observability": FakeModel(script=gather_script),
        "dependency": FakeModel(script=topo_script),
        "change_history": FakeModel(script=change_script),
        "reasoning": FakeModel(structured=reasoner, model="claude-sonnet-5-5"),
        "critic": FakeModel(structured=critic, model="claude-sonnet-5-5"),
        "reporter": FakeModel(structured=reporter),
    }


def test_full_investigation_with_fake_llms_reports_cost_breakdown():
    ledger = UsageLedger()
    workers = llm_workers(load_model_config(env={}), ledger, models=fake_models())
    result = investigate(load_scenario("inc-001"), workers, ledger=ledger, tool_settings=NO_SLEEP)

    assert result.state["stop_reason"] == "accepted"
    report = result.report
    assert report.outcome == "conclusive" and report.root_cause == "checkout-svc v2.14.0"
    assert report.corrective_actions[0].evidence_ids == ["E1"]  # unknown E999 stripped

    by_role = ledger.by_role()
    assert set(by_role) == {"triage", "observability", "dependency", "change_history", "reasoning", "critic", "reporter"}
    assert set(result.by_agent) >= {"triage", "observability", "dependency", "change_history", "reasoning", "critic", "reporter"}
    assert result.usage.cost_usd == pytest.approx(ledger.total().cost_usd)
    assert set(result.by_model) == {"claude-haiku-5-5", "claude-sonnet-5-5"}
    assert result.usage.tool_calls == result.tool_calls
