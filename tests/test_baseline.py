import json

import pytest
from langchain_core.messages import ToolMessage

from investigator.agents.llm_agents import _Caller
from investigator.agents.stubs import StubBaselineModel
from investigator.baseline import LLMBaselineModel, run_baseline
from investigator.llm.config import load_model_config
from investigator.llm.metering import UsageLedger
from investigator.scenario import load_scenario, scenario_ids
from investigator.schemas import BudgetLimits, InvestigationConfig, Usage
from investigator.tools.harness import RetrySettings
from investigator.tools.registry import TOOL_SPECS

from tests.llm_fakes import FakeModel, ai

NO_SLEEP = RetrySettings(max_retries=1, timeout_s=2.0, backoff_base_s=0.0, backoff_cap_s=0.0)


def cfg(**limits):
    return InvestigationConfig(limits=BudgetLimits(**limits))


def seen_calls(messages):
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and str(m.content).startswith("{"):
            out.append(json.loads(m.content.split("... [investigator truncated")[0])["call_id"])
    return out


def submission(call_ids, outcome="conclusive", status="supported", confidence="high", extra_ids=()):
    return {
        "outcome": outcome, "summary": "deploy broke checkout", "root_cause": "checkout-svc v2.14.0",
        "confidence": confidence,
        "hypotheses": [
            {"rank": 1, "statement": "bad deploy", "status": status, "confidence": confidence,
             "supporting": [{"text": "5xx jumped at 14:05", "call_ids": list(call_ids) + list(extra_ids)}]},
            {"rank": 2, "statement": "provider outage", "status": "refuted", "confidence": "medium",
             "contradicting": [{"text": "made-up fact", "call_ids": ["call-9999"]}]},
        ],
        "corrective_actions": [{"rank": 1, "category": "mitigate", "action": "roll back", "rationale": "x",
                                "call_ids": list(call_ids)}],
    }


class Scripted:
    """Baseline model protocol: messages -> (AIMessage, Usage)."""

    def __init__(self, fn):
        self.fn = fn
        self.turns = 0

    def __call__(self, messages):
        self.turns += 1
        return self.fn(messages, self.turns), Usage(llm_calls=1)


def run(model, scenario="inc-001", config=None, **kw):
    return run_baseline(load_scenario(scenario), model, config or cfg(), tool_settings=NO_SLEEP, **kw)


def test_baseline_gets_every_tool_plus_submit():
    fake = FakeModel(script=lambda m: ai())
    LLMBaselineModel(_Caller("baseline", load_model_config(env={})["baseline"], fake, UsageLedger()))
    assert {t["name"] for t in fake.bound_tools} == set(TOOL_SPECS) | {"submit_report"}


def test_grounded_report_strips_fabricated_citations():
    def fn(messages, turn):
        calls = seen_calls(messages)
        if not calls:
            return ai(tool_calls=[{"name": "query_metrics", "args": {"service": "checkout-svc"}},
                                  {"name": "get_changes", "args": {}}])
        return ai(tool_calls=[{"name": "submit_report", "args": submission(calls, extra_ids=["call-4242"])}])

    r = run(Scripted(fn))
    report = r.report
    assert r.state["stop_reason"] == "accepted" and report.outcome == "conclusive"
    assert report.root_cause == "checkout-svc v2.14.0"
    assert [e.source_call_ids for e in report.evidence_index] == [["call-0001", "call-0002"]]
    assert report.hypotheses[0].supporting_evidence == ["E1"]
    assert report.hypotheses[1].contradicting_evidence == []  # fact citing only a fake call is dropped
    assert report.corrective_actions[0].evidence_ids == ["E1"]
    assert any("call-4242" in e for e in r.state["errors"]) and any("call-9999" in e for e in r.state["errors"])


def test_conclusive_claim_without_support_is_reported_inconclusive():
    def fn(messages, turn):
        calls = seen_calls(messages)
        if not calls:
            return ai(tool_calls=[{"name": "get_alerts", "args": {}}])
        return ai(tool_calls=[{"name": "submit_report", "args": submission(calls, status="open", confidence="low")}])

    r = run(Scripted(fn))
    assert r.report.outcome == "inconclusive" and r.report.root_cause is None


def test_tool_budget_forces_submission():
    def fn(messages, turn):
        if any("Budget reached" in str(m.content) for m in messages):
            return ai(tool_calls=[{"name": "submit_report", "args": submission(seen_calls(messages)[:1])}])
        return ai(tool_calls=[{"name": "get_alerts", "args": {}}])

    r = run(Scripted(fn), config=cfg(max_tool_calls=3))
    assert r.tool_calls == 3 and r.usage.tool_calls == 3
    assert r.state["stop_reason"] == "max_tool_calls"
    assert r.report.outcome == "inconclusive"  # forced stop never counts as conclusive


def test_agent_that_never_submits_gets_a_fallback_report():
    model = Scripted(lambda m, t: ai(content="I believe it is the database."))
    r = run(model)
    assert model.turns == 3  # first answer + 2 nudges
    assert r.state["stop_reason"] == "no_progress"
    assert r.report.outcome == "inconclusive" and "did not submit" in r.report.summary


def test_model_failure_is_recovered_into_a_report():
    def fn(messages, turn):
        raise TimeoutError("model timed out")

    r = run(Scripted(fn))
    assert r.report is not None and r.state["stop_reason"] == "no_progress"
    assert any("model call failed" in e for e in r.state["errors"])


def test_invalid_submission_is_sent_back_for_correction():
    def fn(messages, turn):
        if turn == 1:
            return ai(tool_calls=[{"name": "get_alerts", "args": {}}])
        if turn == 2:
            return ai(tool_calls=[{"name": "submit_report", "args": {"outcome": "maybe"}}])
        return ai(tool_calls=[{"name": "submit_report", "args": submission(seen_calls(messages))}])

    r = run(Scripted(fn))
    assert r.state["submission"] is not None and r.report.outcome == "conclusive"


def test_wall_clock_limit():
    ticks = iter(range(0, 100_000, 50))

    def fn(messages, turn):
        if any("Budget reached" in str(m.content) for m in messages):
            return ai(tool_calls=[{"name": "submit_report", "args": submission(seen_calls(messages))}])
        return ai(tool_calls=[{"name": "get_alerts", "args": {}}])

    r = run(Scripted(fn), config=cfg(max_wall_clock_s=120), clock=lambda: float(next(ticks)))
    assert r.state["stop_reason"] == "max_wall_clock"


def test_llm_baseline_is_metered_like_the_multi_agent_system():
    def script(messages):
        calls = seen_calls(messages)
        if not calls:
            return ai(tool_calls=[{"name": "get_alerts", "args": {}}], model="claude-sonnet-5-5")
        return ai(tool_calls=[{"name": "submit_report", "args": submission(calls)}], model="claude-sonnet-5-5")

    ledger = UsageLedger()
    model = LLMBaselineModel(_Caller("baseline", load_model_config(env={})["baseline"], FakeModel(script=script), ledger))
    r = run(model, ledger=ledger)
    assert set(r.by_agent) == {"baseline"} and set(r.by_model) == {"claude-sonnet-5-5"}
    assert r.usage.llm_calls == 2 and r.usage.cost_usd == pytest.approx(ledger.total().cost_usd) and r.usage.cost_usd > 0


@pytest.mark.parametrize("scenario_id", scenario_ids())
def test_stub_baseline_runs_on_every_scenario(scenario_id):
    r = run(StubBaselineModel(), scenario_id)
    assert r.report is not None and r.state["stop_reason"] in {"inconclusive", "accepted"}
    assert r.usage.tool_calls == r.tool_calls
