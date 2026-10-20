import pytest
from langchain_core.messages import ToolMessage

from investigator.agents.llm_agents import (
    LLMCritic,
    LLMGatherer,
    LLMPlanner,
    LLMRefusal,
    ReasoningOutput,
    StructuredOutputError,
    _Caller,
    llm_workers,
    render_tool_result,
)
from investigator.llm.config import PRESETS, PROVIDER_MODELS, TIERS, ModelSpec, load_model_config
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


def test_budget_preset_is_one_tier_below_balanced():
    for role, (tier, _) in PRESETS["budget"].items():
        expected = TIERS[max(0, TIERS.index(PRESETS["balanced"][role][0]) - 1)]
        assert tier == expected, role


def test_provider_switch_fills_the_same_tiers():
    anthropic = load_model_config("budget", env={})
    openai = load_model_config("budget", env={"INVESTIGATOR_PROVIDER": "openai"})
    assert {s.provider for s in openai.values()} == {"openai"}
    for role, (tier, effort) in PRESETS["budget"].items():
        assert anthropic[role].model == PROVIDER_MODELS["anthropic"][tier]
        assert openai[role].model == PROVIDER_MODELS["openai"][tier] and openai[role].effort == effort
    # the explicit argument wins over the env var, and per-role overrides still apply on top
    config = load_model_config("budget", env={"INVESTIGATOR_PROVIDER": "openai", "MODEL_CRITIC": "anthropic:claude-opus-5-5"},
                               provider="anthropic")
    assert config["reasoning"].provider == "anthropic" and config["critic"].model == "claude-opus-5-5"
    with pytest.raises(ValueError):
        load_model_config(env={"INVESTIGATOR_PROVIDER": "nope"})


def test_openai_models_get_mapped_effort_and_no_anthropic_options(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    model = build_chat_model(ModelSpec(provider="openai", model="o3", effort="max"), cache=True)
    # Responses API: Chat Completions rejects tools + reasoning effort on gpt-5.x
    assert model.use_responses_api and model.reasoning == {"effort": "high"} and model.model_name == "o3"


def test_openai_refusal_raises():
    caller = _Caller("critic", load_model_config(env={})["critic"], FakeModel(), UsageLedger())
    refused = ai(content="")
    refused.additional_kwargs["refusal"] = "I can't help with that."
    with pytest.raises(LLMRefusal):
        caller._check(refused)


def test_prices_match_dated_snapshots_but_not_other_models():
    assert cost_usd("o3-2025-04-16", 1_000_000, 1_000_000) == pytest.approx(2.00 + 8.00)
    assert cost_usd("gpt-5.4-2026-03-05", 0, 1_000_000) == pytest.approx(15.00)
    assert cost_usd("o3-pro", 1000, 1000) is None


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


@pytest.mark.parametrize("human_available", [False, True])
def test_critic_is_told_when_no_human_can_answer(human_available):
    seen = []

    def critic(schema, messages):
        seen.append(messages[-1].content)
        return CriticVerdict(verdict="accept")

    spec = load_model_config(env={})["critic"]
    worker = LLMCritic(_Caller("critic", spec, FakeModel(structured=critic), UsageLedger()))
    task = Task(task_id="T9", agent="critic", goal="challenge")
    worker(task, WorkerContext(incident={"title": "x"}, human_available=human_available))
    assert ("never choose needs_human" in seen[0]) is not human_available
    assert "- observability: query_metrics, search_logs" in seen[0]


def change_gatherer(script, tool_calls=(), allowance=4):
    spec = load_model_config(env={})["change_history"]
    model = FakeModel(script=script)
    g = LLMGatherer("change_history", _Caller("change_history", spec, model, UsageLedger()))
    toolbox = ToolBox(ToolHarness(load_scenario("inc-001"), settings=NO_SLEEP), allowance)
    task = Task(task_id="T3", agent="change_history", goal="recent changes to checkout-svc", services=["checkout-svc"])
    ctx = WorkerContext(incident={"title": "x", "reported_at": "2026-09-14T14:30:00Z"}, tool_call_allowance=allowance,
                        tool_calls=list(tool_calls))
    return g, task, ctx, toolbox, model


def test_change_history_always_starts_with_an_unfiltered_sweep():
    g, task, ctx, toolbox, model = change_gatherer(lambda m: ai(tool_calls=[submit(["call-0001"])]))
    outcome = g(task, ctx, toolbox)
    first = toolbox.records()[0]
    assert first.tool == "get_changes" and first.args == {"start": "2026-08-31T14:30:00Z"}  # 14 days before
    assert "cite it as call-0001" in str(model.calls[0][-1].content)
    assert outcome.evidence[0].source_call_ids == ["call-0001"]  # citable like any own call


def test_change_sweep_is_skipped_when_the_team_already_ran_one():
    done = ToolCallRecord(call_id="c9", tool="get_changes", args={"start": "2026-09-01T00:00:00Z"}, ok=True)
    filtered = ToolCallRecord(call_id="c8", tool="get_changes", args={"service": "checkout-svc"}, ok=True)
    g, task, ctx, toolbox, _ = change_gatherer(lambda m: ai(tool_calls=[{"name": "get_infra_events", "args": {}}]) if not
                                               last_call_ids(m) else ai(tool_calls=[submit(last_call_ids(m))]),
                                               tool_calls=[done])
    g(task, ctx, toolbox)
    assert [r.tool for r in toolbox.records()] == ["get_infra_events"]
    g, task, ctx, toolbox, _ = change_gatherer(lambda m: ai(tool_calls=[submit(["call-0001"])]), tool_calls=[filtered])
    g(task, ctx, toolbox)
    assert toolbox.records()[0].tool == "get_changes"  # a service-filtered query is not a sweep


def test_planner_sees_recent_changes_in_all_services():
    seen = []

    def planner(schema, messages):
        seen.append(messages[-1].content)
        return TriagePlan(summary="s", hypotheses=[HypothesisSpec(key="h", statement="x")], tasks=[])

    spec = load_model_config(env={})["triage"]
    toolbox = ToolBox(ToolHarness(load_scenario("inc-001"), settings=NO_SLEEP), 3)
    LLMPlanner(_Caller("triage", spec, FakeModel(structured=planner), UsageLedger()))(
        {"title": "x", "reported_at": "2026-09-14T14:30:00Z"}, toolbox)
    assert [r.tool for r in toolbox.records()] == ["get_topology", "get_changes"]
    assert toolbox.records()[1].args == {"start": "2026-08-31T14:30:00Z"}  # unfiltered: every service
    assert "catalog-svc" in seen[0] and "v2.14.0" in seen[0]  # changes from two different services
    assert "- dependency: get_topology" in seen[0]  # routing: which specialist has which tools


def test_reasoning_reporting_and_baseline_separate_observed_from_inferred():
    from investigator.agents import prompts
    assert all(prompts.OBSERVED_VS_INFERRED in p for p in (prompts.REASONER, prompts.REPORTER, prompts.BASELINE))


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
