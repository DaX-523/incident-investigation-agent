import threading

import pytest
from investigator.checkpoint import memory_checkpointer

from investigator.agents.stubs import ScriptedCritic, make_stub_gatherer, stub_workers
from investigator.runner import Investigation, investigate
from investigator.scenario import load_scenario, scenario_ids
from investigator.schemas import (
    GATHERERS,
    BudgetLimits,
    CriticVerdict,
    EvidenceRequest,
    InvestigationConfig,
)
from investigator.tools.harness import RetrySettings

NO_SLEEP = RetrySettings(max_retries=2, timeout_s=2.0, backoff_base_s=0.0, backoff_cap_s=0.0)


def cfg(mode="eval", **limits):
    return InvestigationConfig(mode=mode, limits=BudgetLimits(**limits))


def run(scenario_id="inc-001", workers=None, config=None, **kw):
    return investigate(load_scenario(scenario_id), workers or stub_workers(), config or cfg(), tool_settings=NO_SLEEP, **kw)


def agents_of(state):
    return [t.agent for t in state["tasks"].values()]


@pytest.mark.parametrize("scenario_id", scenario_ids())
def test_every_scenario_terminates_with_a_report(scenario_id):
    result = run(scenario_id)
    assert result.state["phase"] == "done" and result.report is not None
    assert result.state["stop_reason"] is not None
    assert set(GATHERERS) <= set(agents_of(result.state))  # investigation was not skipped
    assert result.usage.tool_calls == result.tool_calls  # budget matches what the harness actually ran


def test_happy_path_is_accepted_and_conclusive():
    r = run("inc-001")
    assert r.state["stop_reason"] == "accepted"
    assert r.report.outcome == "conclusive"
    assert r.report.evidence_index and all(e.source_call_ids for e in r.report.evidence_index)
    assert agents_of(r.state) == ["observability", "dependency", "change_history", "reasoning", "critic"]


def test_gatherers_run_in_parallel():
    barrier = threading.Barrier(len(GATHERERS), timeout=5)
    workers = stub_workers()

    def wrap(agent):
        inner = make_stub_gatherer(agent)

        def gather(task, context, toolbox):
            barrier.wait()  # deadlocks (BrokenBarrierError) unless all three run concurrently
            return inner(task, context, toolbox)

        return gather

    workers.gatherers = {a: wrap(a) for a in GATHERERS}
    r = run(workers=workers)
    first_round = [t for t in r.state["tasks"].values() if t.agent in GATHERERS and t.round == 1]
    assert all(t.status in {"done", "partial"} for t in first_round), r.state["gaps"]


def test_critic_feedback_loop_is_bounded():
    critic = ScriptedCritic([CriticVerdict(verdict="need_more_evidence",
                                           requests=[EvidenceRequest(agent="observability", goal="dig deeper")])])
    r = run(workers=stub_workers(critic), config=cfg(max_critic_loops=3))
    assert r.state["stop_reason"] == "max_critic_loops"
    assert critic.calls == 3 and r.state["budget"].critic_loops == 3
    # Targeted re-dispatch: rounds 2 and 3 only ran the requested agent.
    later = [t.agent for t in r.state["tasks"].values() if t.round > 1 and t.agent in GATHERERS]
    assert later == ["observability", "observability"]


def test_critic_requested_round_then_accept():
    critic = ScriptedCritic([
        CriticVerdict(verdict="need_more_evidence", requests=[EvidenceRequest(agent="change_history", goal="recent deploys")]),
        CriticVerdict(verdict="accept"),
    ])
    r = run(workers=stub_workers(critic))
    assert r.state["stop_reason"] == "accepted" and critic.calls == 2


def test_crashing_worker_is_retried_then_recorded_as_failed():
    attempts = []
    workers = stub_workers()

    def broken(task, context, toolbox):
        attempts.append(task.task_id)
        raise RuntimeError("LLM returned garbage")

    workers.gatherers["dependency"] = broken
    r = run(workers=workers, config=cfg(max_worker_attempts=2))
    assert r.report is not None
    assert attempts == ["T2", "T2"]
    assert r.state["tasks"]["T2"].status == "failed"
    assert any(g.kind == "worker_failure" and "garbage" in g.description for g in r.state["gaps"])


def test_tool_failures_surface_as_gaps():
    r = run("inc-008")
    assert any(g.kind == "tool_failure" and "search_logs" in g.description for g in r.state["gaps"])
    assert any("search_logs" in g for g in r.report.gaps)


def test_planner_failure_falls_back_to_default_coverage():
    workers = stub_workers()

    def broken_planner(incident, toolbox):
        raise TimeoutError("planner model timed out")

    workers.planner = broken_planner
    r = run(workers=workers)
    assert any("planner failed" in e for e in r.state["errors"])
    assert r.state["hypotheses"]["H1"].origin == "reporter"
    assert set(GATHERERS) <= set(agents_of(r.state))
    assert r.report is not None


def test_reporter_failure_falls_back_to_deterministic_report():
    workers = stub_workers()

    def broken_reporter(state):
        raise ValueError("bad json")

    workers.reporter = broken_reporter
    r = run(workers=workers)
    assert r.report is not None and r.report.hypotheses
    assert any("reporter failed" in e for e in r.state["errors"])


def test_needs_human_in_eval_mode_reports_inconclusive_with_questions():
    critic = ScriptedCritic([CriticVerdict(verdict="needs_human", questions_for_human=["Was a manual change made?"])])
    r = run(workers=stub_workers(critic), config=cfg(mode="eval"))
    assert r.state["stop_reason"] == "needs_human"
    assert r.report.outcome == "inconclusive"
    assert r.report.questions_for_human == ["Was a manual change made?"]


def test_needs_human_interactive_interrupts_and_resumes_from_checkpoint():
    critic = ScriptedCritic([
        CriticVerdict(verdict="needs_human", questions_for_human=["Was a manual change made?"]),
        CriticVerdict(verdict="accept"),
    ])
    inv = Investigation(load_scenario("inc-001"), stub_workers(critic), cfg(mode="interactive"),
                        checkpointer=memory_checkpointer(), tool_settings=NO_SLEEP)
    try:
        paused = inv.start()
        assert paused.interrupted and paused.report is None
        assert paused.state["__interrupt__"][0].value["questions"] == ["Was a manual change made?"]
        done = inv.resume("Yes, a firewall rule was changed at 14:00")
    finally:
        inv.close()
    assert not done.interrupted and done.report is not None
    assert done.state["human_inputs"] == ["Yes, a firewall rule was changed at 14:00"]
    assert done.state["stop_reason"] == "accepted"


def test_wall_clock_limit_stops_the_run():
    ticks = iter(range(0, 10_000, 100))  # every clock read advances 100s
    r = run(config=cfg(max_wall_clock_s=150), clock=lambda: float(next(ticks)))
    assert r.state["stop_reason"] == "max_wall_clock"
    assert r.report is not None


def test_tool_call_budget_stops_the_run():
    r = run(config=cfg(max_tool_calls=4))
    assert r.state["stop_reason"] == "max_tool_calls"


def test_per_task_tool_allowance_is_enforced():
    workers = stub_workers()
    seen = []

    def greedy(task, context, toolbox):
        results = [toolbox.call("get_alerts", {}) for _ in range(10)]
        seen.append((context.tool_call_allowance, [r.error.kind if r.error else "ok" for r in results]))
        return make_stub_gatherer("observability")(task, context, toolbox)

    workers.gatherers["observability"] = greedy
    r = run(workers=workers, config=cfg(max_tool_calls_per_task=3))
    allowance, kinds = seen[0]
    assert allowance == 3 and kinds.count("ok") == 3 and kinds.count("budget_exhausted") == 7
    obs = [t for t in r.state["tasks"].values() if t.agent == "observability"][0]
    assert len([o for o in r.state["outcomes"] if o.task_id == obs.task_id][0].tool_calls) == 3


def test_worker_payload_contains_no_fault_config():
    workers = stub_workers()
    contexts = []

    def spy(task, context, toolbox):
        contexts.append(context.model_dump_json())
        return make_stub_gatherer("observability")(task, context, toolbox)

    workers.gatherers["observability"] = spy
    scenario = load_scenario("inc-009")
    investigate(scenario, workers, cfg(), tool_settings=NO_SLEEP)
    notes = [f.note for f in scenario.tool_faults]
    assert contexts and not any(n in c for n in notes for c in contexts)
    assert all("tool_faults" not in c for c in contexts)


def test_checkpoint_resume_uses_strict_allowlist(caplog):
    critic = ScriptedCritic([CriticVerdict(verdict="needs_human", questions_for_human=["q"]), CriticVerdict(verdict="accept")])
    inv = Investigation(load_scenario("inc-001"), stub_workers(critic), cfg(mode="interactive"),
                        checkpointer=memory_checkpointer(), tool_settings=NO_SLEEP)
    try:
        inv.start()
        with caplog.at_level("WARNING"):
            done = inv.resume("ok")
    finally:
        inv.close()
    assert done.report is not None
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]
