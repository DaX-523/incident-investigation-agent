import random

import pytest
from pydantic import ValidationError

from investigator.orchestrator.manager import (
    apply_outcomes,
    apply_triage,
    check_limits,
    decide,
    evidence_gate,
    init_state,
    ranked_hypotheses,
    ready_tasks,
    step,
)
from investigator.schemas import (
    BudgetLimits,
    CriticVerdict,
    EvidenceDraft,
    EvidenceRequest,
    HypothesisAssessment,
    HypothesisLink,
    HypothesisSpec,
    InvestigationConfig,
    Outcome,
    TaskSpec,
    ToolCallRecord,
    TriagePlan,
    Usage,
    failed_outcome,
)

T0 = 1_000_000.0


def merged(state, update):
    return {**state, **update}


def new_state(mode="eval", **limits):
    return init_state({"title": "checkout errors"}, InvestigationConfig(mode=mode, limits=BudgetLimits(**limits)), now=T0)


def plan(tasks=None, hypotheses=None):
    return TriagePlan(
        summary="checkout failing",
        affected_services=["checkout-svc"],
        hypotheses=hypotheses or [
            HypothesisSpec(key="h_report", statement="Network outage", origin="reporter"),
            HypothesisSpec(key="h_deploy", statement="Bad deploy"),
        ],
        tasks=tasks if tasks is not None else [
            TaskSpec(key="a", agent="observability", goal="error rates", hypothesis_keys=["h_report"]),
            TaskSpec(key="b", agent="change_history", goal="recent deploys", hypothesis_keys=["h_deploy"]),
            TaskSpec(key="c", agent="dependency", goal="blast radius", depends_on=["a"]),
        ],
    )


def triaged(**kw):
    state = new_state(**kw)
    return merged(state, apply_triage(state, plan()))


def call(cid, tool="query_metrics", ok=True, notes=()):
    return ToolCallRecord(call_id=cid, tool=tool, ok=ok, error_kind=None if ok else "timeout",
                          error_message=None if ok else "timed out", quality_notes=list(notes))


def dispatch(state, now=T0):
    d = decide(state, now)
    assert d.action == "dispatch"
    return merged(state, d.update), d.tasks


def deliver(state, *outcomes):
    return merged(state, {"outcomes": state["outcomes"] + list(outcomes)})


def gatherer_outcome(task, call_ids=("c1",), link="H1", stance="supports", status="ok"):
    return Outcome(
        task_id=task.task_id, agent=task.agent, status=status,
        tool_calls=[call(c) for c in call_ids],
        evidence=[EvidenceDraft(summary=f"{task.agent} saw something", source_call_ids=list(call_ids),
                                links=[HypothesisLink(hypothesis_id=link, stance=stance)])],
        usage=Usage(llm_calls=1, input_tokens=100, output_tokens=20),
    )


def run_gather_round(state):
    """Dispatch ready gatherers, deliver evidence for each, apply. Returns state after apply."""
    state, tasks = dispatch(state)
    for i, t in enumerate(tasks):
        state = deliver(state, gatherer_outcome(t, call_ids=(f"{t.task_id}-c{i}",)))
    return merged(state, apply_outcomes(state))


# --- triage ------------------------------------------------------------------------


def test_triage_assigns_ids_and_resolves_dependencies():
    state = triaged()
    assert list(state["hypotheses"]) == ["H1", "H2"]
    assert state["hypotheses"]["H1"].origin == "reporter"
    assert [t.task_id for t in state["tasks"].values()] == ["T1", "T2", "T3"]
    assert state["tasks"]["T3"].depends_on == ["T1"]
    assert state["tasks"]["T1"].hypothesis_ids == ["H1"]
    assert state["phase"] == "gather" and state["budget"].rounds == 1


def test_triage_adds_missing_gatherers_so_investigation_cannot_be_skipped():
    state = new_state()
    state = merged(state, apply_triage(state, plan(tasks=[TaskSpec(key="a", agent="observability", goal="x")])))
    assert {t.agent for t in state["tasks"].values()} == {"observability", "dependency", "change_history"}


def test_triage_caps_tasks_and_drops_bad_dependencies():
    state = new_state(max_tasks_per_round=3)
    specs = [TaskSpec(key=str(i), agent="observability", goal="x", depends_on=["nope"]) for i in range(5)]
    state = merged(state, apply_triage(state, plan(tasks=specs)))
    assert len(state["tasks"]) == 3  # 1 kept + 2 coverage tasks
    assert any("max_tasks_per_round" in e for e in state["errors"])
    assert any("unknown dependency" in e for e in state["errors"])


def test_triage_breaks_dependency_cycles():
    state = new_state()
    specs = [
        TaskSpec(key="a", agent="observability", goal="x", depends_on=["b"]),
        TaskSpec(key="b", agent="dependency", goal="y", depends_on=["a"]),
        TaskSpec(key="c", agent="change_history", goal="z"),
    ]
    state = merged(state, apply_triage(state, plan(tasks=specs)))
    assert any("cycle" in e for e in state["errors"])
    assert ready_tasks(state)  # something is runnable, no deadlock


# --- dispatch & dependencies --------------------------------------------------------


def test_dependent_task_waits_for_its_dependency():
    state = triaged()
    state, first = dispatch(state)
    assert [t.task_id for t in first] == ["T1", "T2"]
    assert all(state["tasks"][t.task_id].status == "dispatched" for t in first)
    state = deliver(state, *(gatherer_outcome(t, call_ids=(f"x{t.task_id}",)) for t in first))
    state = merged(state, apply_outcomes(state))
    state, second = dispatch(state)
    assert [t.task_id for t in second] == ["T3"]


def test_failed_dependency_still_unblocks_dependent_task():
    state = triaged()
    state, first = dispatch(state)
    state = deliver(state, failed_outcome(first[0], RuntimeError("boom")),
                    gatherer_outcome(first[1], call_ids=("c2",)))
    state = merged(state, apply_outcomes(state))
    assert state["tasks"]["T1"].status == "failed"
    assert any(g.kind == "worker_failure" and "boom" in g.description for g in state["gaps"])
    assert [t.task_id for t in ready_tasks(state)] == ["T3"]


# --- applying outcomes ----------------------------------------------------------------


def test_apply_is_deterministic_regardless_of_arrival_order():
    base = triaged()
    base, tasks = dispatch(base)
    outcomes = [gatherer_outcome(t, call_ids=(f"c-{t.task_id}",)) for t in tasks]
    results = []
    for seed in range(5):
        shuffled = outcomes[:]
        random.Random(seed).shuffle(shuffled)
        s = deliver(base, *shuffled)
        results.append([(e.id, e.task_id) for e in apply_outcomes(s)["evidence"]])
    assert all(r == results[0] for r in results)
    assert results[0] == [("E1", "T1"), ("E2", "T2")]


def test_ungrounded_evidence_is_dropped():
    state = triaged()
    state, tasks = dispatch(state)
    bad = Outcome(task_id=tasks[0].task_id, agent=tasks[0].agent, status="ok", tool_calls=[call("real")],
                  evidence=[EvidenceDraft(summary="made up", source_call_ids=["invented"])])
    state = merged(deliver(state, bad), {})
    state = merged(state, apply_outcomes(state))
    assert state["evidence"] == []
    assert any("ungrounded" in e for e in state["errors"])


def test_tool_failures_and_quality_notes_become_gaps_automatically():
    state = triaged()
    state, tasks = dispatch(state)
    out = Outcome(task_id=tasks[0].task_id, agent=tasks[0].agent, status="partial",
                  tool_calls=[call("c1", tool="search_logs", ok=False), call("c2", notes=["returned data outside requested range"])])
    state = merged(deliver(state, out), {})
    state = merged(state, apply_outcomes(state))
    kinds = sorted(g.kind for g in state["gaps"])
    assert kinds == ["data_quality", "tool_failure"]
    assert state["tasks"][tasks[0].task_id].status == "partial"
    assert state["budget"].usage.tool_calls == 2


def test_duplicate_and_stale_outcomes_are_ignored():
    state = triaged()
    state, tasks = dispatch(state)
    o = gatherer_outcome(tasks[0])
    state = deliver(state, o, o)
    state = merged(state, apply_outcomes(state))
    assert len(state["evidence"]) == 1
    assert any("ignored outcome" in e for e in state["errors"])
    # Re-applying with no new mailbox entries changes nothing.
    again = apply_outcomes(state)
    assert again["evidence"] == state["evidence"] and again["outcomes_applied"] == state["outcomes_applied"]


def test_evidence_links_to_unknown_hypotheses_are_dropped():
    state = triaged()
    state, tasks = dispatch(state)
    state = deliver(state, gatherer_outcome(tasks[0], link="H99"))
    state = merged(state, apply_outcomes(state))
    assert state["evidence"][0].links == []


def test_reasoning_assessments_rank_and_strip_unknown_evidence():
    state = run_gather_round(triaged())
    state, (reason_task,) = dispatch(merged(state, {"phase": "gather"}))  # T3 next
    state = deliver(state, gatherer_outcome(reason_task, call_ids=("c9",)))
    state = merged(state, apply_outcomes(state))
    d = decide(state, T0)
    assert d.tasks[0].agent == "reasoning"
    state = merged(state, d.update)
    outcome = Outcome(task_id=d.tasks[0].task_id, agent="reasoning", status="ok", assessments=[
        HypothesisAssessment(hypothesis_id="H2", statement="Bad deploy", rank=1, status="supported", confidence="high",
                             supporting=["E2", "E77"], rationale="deploy precedes errors"),
        HypothesisAssessment(hypothesis_id="H1", statement="Network outage", rank=1, status="refuted", confidence="medium",
                             contradicting=["E1"], rationale="packet loss flat"),
        HypothesisAssessment(statement="Cert expiry", rank=3, status="open", confidence="low", rationale="untested"),
    ])
    state = merged(deliver(state, outcome), {})
    state = merged(state, apply_outcomes(state))
    ranked = ranked_hypotheses(state)
    assert [h.id for h in ranked] == ["H1", "H2", "H3"]  # tie on rank 1 broken by id
    assert state["hypotheses"]["H2"].supporting == ["E2"]
    assert any("unknown evidence" in e for e in state["errors"])
    assert state["hypotheses"]["H3"].origin == "reasoning"


def test_outcome_payload_validation():
    with pytest.raises(ValidationError):
        Outcome(task_id="T1", agent="critic", status="ok")
    Outcome(task_id="T1", agent="critic", status="failed")  # failed outcomes need no payload


# --- full routing loop -----------------------------------------------------------------


def to_critique(state, verdict):
    """Drive gather -> reason -> critic with a given verdict; return state before routing the verdict."""
    while True:
        d = decide(state, T0)
        if d.tasks and d.tasks[0].agent == "reasoning":
            break
        state = merged(state, d.update)
        for i, t in enumerate(d.tasks):
            state = deliver(state, gatherer_outcome(t, call_ids=(f"{t.task_id}-{i}",), link="H1"))
        state = merged(state, apply_outcomes(state))
    state = merged(state, d.update)
    assessments = [HypothesisAssessment(hypothesis_id="H1", statement="x", rank=1, status="supported",
                                        confidence="high", supporting=[state["evidence"][0].id], rationale="r")]
    state = deliver(state, Outcome(task_id=d.tasks[0].task_id, agent="reasoning", status="ok", assessments=assessments))
    state = merged(state, apply_outcomes(state))
    d = decide(state, T0)
    assert d.tasks[0].agent == "critic"
    state = merged(state, d.update)
    state = deliver(state, Outcome(task_id=d.tasks[0].task_id, agent="critic", status="ok", verdict=verdict))
    return state


def test_accept_with_evidence_goes_to_report():
    state = to_critique(triaged(gate_top_k=1), CriticVerdict(verdict="accept"))
    d = step(state, T0)
    assert d.action == "report" and d.stop_reason == "accepted"
    assert d.update["phase"] == "report"


def test_inconclusive_verdict_reports_inconclusive():
    state = to_critique(triaged(gate_top_k=1), CriticVerdict(verdict="inconclusive"))
    assert step(state, T0).stop_reason == "inconclusive"


def test_need_more_evidence_creates_targeted_round():
    req = EvidenceRequest(agent="observability", goal="check per-AZ latency", hypothesis_ids=["H1", "H404"])
    state = to_critique(triaged(gate_top_k=1), CriticVerdict(verdict="need_more_evidence", requests=[req]))
    d = step(state, T0)
    assert d.action == "dispatch" and len(d.tasks) == 1
    t = d.tasks[0]
    assert t.agent == "observability" and t.hypothesis_ids == ["H1"] and t.round == 2
    assert d.update["budget"].critic_loops == 1 and d.update["budget"].rounds == 2


def test_critic_loops_are_bounded():
    req = EvidenceRequest(agent="observability", goal="more")
    state = to_critique(triaged(gate_top_k=1, max_critic_loops=1), CriticVerdict(verdict="need_more_evidence", requests=[req]))
    d = step(state, T0)
    assert d.action == "report" and d.stop_reason == "max_critic_loops"


def test_gate_failure_triggers_gap_tasks_then_reports_at_limit():
    # Critic accepts, but H2 (top-2) has no evidence: gate sends targeted tasks.
    state = to_critique(triaged(gate_top_k=2), CriticVerdict(verdict="accept"))
    state = merged(state, apply_outcomes(state))
    state = merged(state, {"hypotheses": {**state["hypotheses"],
                                          "H2": state["hypotheses"]["H2"].model_copy(update={"rank": 2})}})
    gate = evidence_gate(state)
    assert not gate.passed and gate.missing_hypotheses == ["H2"]
    d = decide(state, T0)
    assert d.action == "dispatch" and all("H2" in t.goal for t in d.tasks)


def test_needs_human_interactive_vs_eval():
    v = CriticVerdict(verdict="needs_human", questions_for_human=["Was there a manual change?"])
    assert step(to_critique(triaged(mode="interactive"), v), T0).action == "human"
    d = step(to_critique(triaged(mode="eval"), v), T0)
    assert d.action == "report" and d.stop_reason == "needs_human"


def test_failed_critic_falls_back_to_gate():
    state = to_critique(triaged(gate_top_k=1), CriticVerdict(verdict="accept"))
    critic = [t for t in state["tasks"].values() if t.agent == "critic"][0]
    state = merged(state, {"outcomes": state["outcomes"][:-1] + [failed_outcome(critic, "LLM parse error")]})
    d = step(state, T0)
    assert d.action == "report" and d.stop_reason == "accepted"


def test_reasoning_with_no_ranking_reports_no_progress():
    state = run_gather_round(triaged())
    state = run_gather_round(state)  # dependent task T3
    d = decide(state, T0)
    assert d.tasks[0].agent == "reasoning"
    state = merged(state, d.update)
    state = deliver(state, failed_outcome(d.tasks[0], "model returned invalid JSON"))
    d = step(state, T0)
    assert d.action == "report" and d.stop_reason == "no_progress"


# --- limits ---------------------------------------------------------------------------------


def test_limits():
    state = triaged(max_tool_calls=2, max_tokens=10_000, max_wall_clock_s=60)
    assert check_limits(state, T0) is None
    assert check_limits(state, T0 + 61) == "max_wall_clock"
    budget = state["budget"].model_copy(update={"usage": Usage(tool_calls=2)})
    assert check_limits(merged(state, {"budget": budget}), T0) == "max_tool_calls"
    budget = state["budget"].model_copy(update={"usage": Usage(input_tokens=9_000, output_tokens=1_000)})
    assert check_limits(merged(state, {"budget": budget}), T0) == "max_tokens"


def test_limit_hit_during_gather_stops_dispatch():
    state = triaged(max_tool_calls=1)
    state, tasks = dispatch(state)
    state = deliver(state, gatherer_outcome(tasks[0], call_ids=("a", "b")))
    d = step(state, T0)
    assert d.action == "report" and d.stop_reason == "max_tool_calls"


def test_wall_clock_limit_routes_to_report():
    state = triaged(max_wall_clock_s=5)
    d = decide(state, T0 + 10)
    assert d.action == "report" and d.stop_reason == "max_wall_clock"


def test_state_input_is_never_mutated():
    state = triaged()
    snapshot = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in state.items()}
    state2, tasks = dispatch(state)
    step(deliver(state2, gatherer_outcome(tasks[0])), T0)
    assert all(state[k] == snapshot[k] for k in ("tasks", "evidence", "hypotheses", "gaps", "errors"))
    assert state["budget"] == snapshot["budget"]
