"""Deterministic half of the orchestrator: the single writer of investigation state.

Pure functions: each takes a state snapshot and returns a partial update dict
(never mutates its input). Outcomes are applied in task-id order, so parallel
workers finishing in any order produce the same state.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from investigator.schemas import (
    GATHERERS,
    AgentName,
    Budget,
    CriticVerdict,
    Evidence,
    EvidenceRequest,
    Gap,
    Hypothesis,
    InvestigationConfig,
    InvestigationState,
    Outcome,
    StopReason,
    Task,
    TriagePlan,
)

RESOLVED = {"done", "partial", "failed", "skipped"}
_STATUS = {"ok": "done", "partial": "partial", "failed": "failed"}

_SWEEP_GOALS: dict[str, str] = {
    "observability": "Sweep metrics, logs, traces, alerts and infra events for the affected services in the incident window.",
    "dependency": "Map the affected services' upstream callers and downstream dependencies and where the symptoms sit in that chain.",
    "change_history": "List deploys, config/flag changes, migrations and other changes before and during the incident, and check past incidents and runbooks.",
}


def _num(identifier: str) -> int:
    match = re.search(r"(\d+)$", identifier)
    return int(match.group(1)) if match else 0


def _sorted_tasks(tasks: dict[str, Task]) -> list[Task]:
    return sorted(tasks.values(), key=lambda t: _num(t.task_id))


# --- init / triage ----------------------------------------------------------------


def init_state(incident: dict[str, Any], config: InvestigationConfig | None = None, now: float | None = None) -> InvestigationState:
    return InvestigationState(
        incident=incident,
        config=config or InvestigationConfig(),
        outcomes=[],
        outcomes_applied=0,
        triage=None,
        tasks={},
        evidence=[],
        hypotheses={},
        verdicts=[],
        gaps=[],
        errors=[],
        budget=Budget(started_at=time.time() if now is None else now),
        phase="triage",
        stop_reason=None,
        report=None,
    )


def apply_triage(state: InvestigationState, plan: TriagePlan) -> dict[str, Any]:
    """Turn the planner's proposal into hypotheses and round-1 tasks.

    Guarantees every gatherer gets at least one task so the investigation can't be skipped.
    """
    limits = state["config"].limits
    errors: list[str] = []
    hypotheses: dict[str, Hypothesis] = {}
    hyp_ids: dict[str, str] = {}
    for spec in plan.hypotheses:
        hid = f"H{len(hypotheses) + 1}"
        hyp_ids[spec.key] = hid
        hypotheses[hid] = Hypothesis(id=hid, statement=spec.statement, origin=spec.origin)

    covered = {t.agent for t in plan.tasks}
    missing = [g for g in GATHERERS if g not in covered]
    room = max(0, limits.max_tasks_per_round - len(missing))
    proposed = plan.tasks[:room]
    if len(plan.tasks) > room:
        errors.append(f"triage proposed {len(plan.tasks)} tasks; kept {room} (max_tasks_per_round)")

    tasks: dict[str, Task] = {}
    task_ids: dict[str, str] = {}
    for spec in proposed:
        task_ids[spec.key] = f"T{len(task_ids) + 1}"
    for spec in proposed:
        deps = []
        for key in spec.depends_on:
            if key in task_ids and key != spec.key:
                deps.append(task_ids[key])
            else:
                errors.append(f"task {spec.key!r}: dropped unknown dependency {key!r}")
        linked = [hyp_ids[k] for k in spec.hypothesis_keys if k in hyp_ids]
        tid = task_ids[spec.key]
        tasks[tid] = Task(task_id=tid, agent=spec.agent, goal=spec.goal, hypothesis_ids=linked,
                          services=spec.services, depends_on=deps, round=1)
    for agent in missing:
        tid = f"T{len(tasks) + 1}"
        tasks[tid] = Task(task_id=tid, agent=agent, goal=_SWEEP_GOALS[agent], hypothesis_ids=list(hypotheses),
                          services=plan.affected_services, round=1)

    errors.extend(_break_cycles(tasks))
    budget = state["budget"].model_copy(update={"rounds": 1})
    return {
        "triage": plan,
        "hypotheses": hypotheses,
        "tasks": tasks,
        "budget": budget,
        "errors": state["errors"] + errors,
        "phase": "gather",
    }


def _break_cycles(tasks: dict[str, Task]) -> list[str]:
    """Drop dependencies that would create a cycle (mutates the freshly built task dict)."""
    errors = []
    state: dict[str, int] = {}

    def visit(tid: str) -> None:
        state[tid] = 1
        task = tasks[tid]
        kept = []
        for dep in task.depends_on:
            if state.get(dep) == 1:
                errors.append(f"dependency cycle {tid} -> {dep}; dependency dropped")
                continue
            if state.get(dep) is None:
                visit(dep)
            kept.append(dep)
        tasks[tid] = task.model_copy(update={"depends_on": kept})
        state[tid] = 2

    for tid in list(tasks):
        if state.get(tid) is None:
            visit(tid)
    return errors


# --- dispatch ---------------------------------------------------------------------


def ready_tasks(state: InvestigationState) -> list[Task]:
    tasks = state["tasks"]
    return [
        t
        for t in _sorted_tasks(tasks)
        if t.status == "pending"
        and t.agent in GATHERERS
        and all(tasks[d].status in RESOLVED for d in t.depends_on if d in tasks)
    ]


def _mark(tasks: dict[str, Task], ids: list[str], status: str) -> dict[str, Task]:
    updated = dict(tasks)
    for tid in ids:
        updated[tid] = updated[tid].model_copy(update={"status": status})
    return updated


def _new_task(tasks: dict[str, Task], agent: AgentName, goal: str, round_: int, **extra: Any) -> Task:
    tid = f"T{max((_num(t) for t in tasks), default=0) + 1}"
    return Task(task_id=tid, agent=agent, goal=goal, round=round_, **extra)


# --- applying outcomes ----------------------------------------------------------------


def apply_outcomes(state: InvestigationState) -> dict[str, Any]:
    """Consume new mailbox entries, in task-id order, into orchestrator-owned state."""
    applied = state.get("outcomes_applied", 0)
    new = sorted(state["outcomes"][applied:], key=lambda o: (_num(o.task_id), o.agent))

    tasks = dict(state["tasks"])
    evidence = list(state["evidence"])
    hypotheses = dict(state["hypotheses"])
    verdicts = list(state["verdicts"])
    gaps = list(state["gaps"])
    errors = list(state["errors"])
    budget = state["budget"]
    seen = {(frozenset(e.source_call_ids), e.summary.strip().lower()) for e in evidence}

    for outcome in new:
        task = tasks.get(outcome.task_id)
        if task is None or task.status != "dispatched" or task.agent != outcome.agent:
            errors.append(f"ignored outcome for {outcome.task_id} ({outcome.agent}): no matching dispatched task")
            continue

        tool_calls = max(outcome.usage.tool_calls, len(outcome.tool_calls))
        budget = budget.model_copy(update={"usage": budget.usage + outcome.usage.model_copy(update={"tool_calls": tool_calls})})

        calls = {c.call_id: c for c in outcome.tool_calls}
        for c in outcome.tool_calls:
            if not c.ok:
                gaps.append(Gap(kind="tool_failure", agent=outcome.agent, task_id=task.task_id, call_id=c.call_id,
                                description=f"{c.tool} {c.args} failed: {c.error_kind}: {c.error_message}"))
            for note in c.quality_notes:
                gaps.append(Gap(kind="data_quality", agent=outcome.agent, task_id=task.task_id, call_id=c.call_id,
                                description=f"{c.tool} {c.args}: {note}"))
        for text in outcome.gaps:
            gaps.append(Gap(kind="not_checked", agent=outcome.agent, task_id=task.task_id, description=text))
        if outcome.status == "failed":
            for text in outcome.errors or ["worker failed"]:
                gaps.append(Gap(kind="worker_failure", agent=outcome.agent, task_id=task.task_id, description=text))

        for draft in outcome.evidence:
            unknown = [cid for cid in draft.source_call_ids if cid not in calls]
            if unknown:
                errors.append(f"{task.task_id}: dropped ungrounded evidence (unknown call ids {unknown}): {draft.summary[:80]!r}")
                continue
            key = (frozenset(draft.source_call_ids), draft.summary.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            links = []
            for link in draft.links:
                if link.hypothesis_id in hypotheses:
                    links.append(link)
                else:
                    errors.append(f"{task.task_id}: evidence linked to unknown hypothesis {link.hypothesis_id}")
            evidence.append(Evidence(
                id=f"E{len(evidence) + 1}",
                summary=draft.summary,
                source_call_ids=draft.source_call_ids,
                tools=sorted({calls[c].tool for c in draft.source_call_ids}),
                agent=outcome.agent,
                task_id=task.task_id,
                round=task.round,
                services=draft.services,
                links=links,
                observed_at=draft.observed_at,
                quality_notes=[n for c in draft.source_call_ids for n in calls[c].quality_notes],
            ))

        if outcome.assessments is not None:
            hypotheses, assess_errors = _apply_assessments(hypotheses, outcome, {e.id for e in evidence})
            errors.extend(assess_errors)
        if outcome.verdict is not None:
            verdicts.append(outcome.verdict)
        if outcome.agent == "critic":
            budget = budget.model_copy(update={"critic_loops": budget.critic_loops + 1})

        tasks[task.task_id] = task.model_copy(update={"status": _STATUS[outcome.status]})

    return {
        "outcomes_applied": len(state["outcomes"]),
        "tasks": tasks,
        "evidence": evidence,
        "hypotheses": hypotheses,
        "verdicts": verdicts,
        "gaps": gaps,
        "errors": errors,
        "budget": budget,
    }


def _apply_assessments(
    hypotheses: dict[str, Hypothesis], outcome: Outcome, evidence_ids: set[str]
) -> tuple[dict[str, Hypothesis], list[str]]:
    errors: list[str] = []
    updated = dict(hypotheses)
    assessed: list[tuple[int, str]] = []
    for a in outcome.assessments or []:
        def known(ids: list[str]) -> list[str]:
            bad = [i for i in ids if i not in evidence_ids]
            if bad:
                errors.append(f"reasoning cited unknown evidence {bad} for {a.hypothesis_id or a.statement[:40]!r}")
            return [i for i in ids if i in evidence_ids]

        hid = a.hypothesis_id
        if hid is None or hid not in updated:
            if hid is not None:
                errors.append(f"reasoning referenced unknown hypothesis {hid}; added as new")
            hid = f"H{max((_num(h) for h in updated), default=0) + 1}"
            updated[hid] = Hypothesis(id=hid, statement=a.statement, origin="reasoning")
        updated[hid] = updated[hid].model_copy(update={
            "status": a.status,
            "confidence": a.confidence,
            "supporting": known(a.supporting),
            "contradicting": known(a.contradicting),
            "assumptions": a.assumptions,
            "rationale": a.rationale,
        })
        assessed.append((a.rank, hid))
    # Normalise ranks to 1..n in the order reasoning gave (ties broken by id).
    for rank, (_, hid) in enumerate(sorted(assessed, key=lambda p: (p[0], _num(p[1]))), start=1):
        updated[hid] = updated[hid].model_copy(update={"rank": rank})
    return updated, errors


# --- limits and gate --------------------------------------------------------------------


def check_limits(state: InvestigationState, now: float | None = None) -> StopReason | None:
    limits = state["config"].limits
    budget = state["budget"]
    now = time.time() if now is None else now
    if budget.usage.tool_calls >= limits.max_tool_calls:
        return "max_tool_calls"
    if budget.usage.tokens >= limits.max_tokens:
        return "max_tokens"
    if now - budget.started_at >= limits.max_wall_clock_s:
        return "max_wall_clock"
    return None


@dataclass
class GateResult:
    passed: bool
    missing_hypotheses: list[str] = field(default_factory=list)
    unchecked_agents: list[str] = field(default_factory=list)


def ranked_hypotheses(state: InvestigationState) -> list[Hypothesis]:
    return sorted(state["hypotheses"].values(), key=lambda h: (h.rank is None, h.rank or 0, _num(h.id)))


def evidence_gate(state: InvestigationState) -> GateResult:
    """Report only when every gatherer ran and each top hypothesis has evidence for or against it."""
    k = state["config"].limits.gate_top_k
    linked: dict[str, int] = {}
    for e in state["evidence"]:
        for link in e.links:
            if link.stance != "neutral":
                linked[link.hypothesis_id] = linked.get(link.hypothesis_id, 0) + 1
    missing = [
        h.id
        for h in ranked_hypotheses(state)[:k]
        if not (h.supporting or h.contradicting or linked.get(h.id))
    ]
    attempted = {t.agent for t in state["tasks"].values() if t.status in {"done", "partial", "failed"}}
    unchecked = [g for g in GATHERERS if g not in attempted]
    return GateResult(passed=not missing and not unchecked, missing_hypotheses=missing, unchecked_agents=unchecked)


# --- routing ------------------------------------------------------------------------------


Action = Literal["dispatch", "human", "report"]


@dataclass
class Decision:
    action: Action
    update: dict[str, Any]
    tasks: list[Task] = field(default_factory=list)
    stop_reason: StopReason | None = None


def _report(state: InvestigationState, reason: StopReason, update: dict[str, Any] | None = None) -> Decision:
    update = dict(update or {})
    update.update(phase="report", stop_reason=reason)
    return Decision(action="report", update=update, stop_reason=reason)


def _dispatch(state: InvestigationState, tasks: list[Task], phase: str, update: dict[str, Any] | None = None) -> Decision:
    update = dict(update or {})
    all_tasks = dict(update.get("tasks", state["tasks"]))
    for t in tasks:
        all_tasks[t.task_id] = t.model_copy(update={"status": "dispatched"})
    update.update(tasks=all_tasks, phase=phase)
    return Decision(action="dispatch", update=update, tasks=[all_tasks[t.task_id] for t in tasks])


def _merged(state: InvestigationState, update: dict[str, Any]) -> InvestigationState:
    return InvestigationState(**{**state, **update})


def _single(state: InvestigationState, agent: Literal["reasoning", "critic"], goal: str) -> Task:
    return _new_task(state["tasks"], agent, goal, state["budget"].rounds)


def _tasks_from_requests(state: InvestigationState, requests: list[EvidenceRequest]) -> list[Task]:
    limits = state["config"].limits
    tasks = dict(state["tasks"])
    created = []
    round_ = state["budget"].rounds + 1
    for req in requests[: limits.max_tasks_per_round]:
        task = _new_task(tasks, req.agent, req.goal, round_,
                         hypothesis_ids=[h for h in req.hypothesis_ids if h in state["hypotheses"]],
                         services=req.services)
        tasks[task.task_id] = task
        created.append(task)
    return created


def _tasks_for_gate(state: InvestigationState, gate: GateResult) -> list[Task]:
    requests = [EvidenceRequest(agent=a, goal=_SWEEP_GOALS[a]) for a in gate.unchecked_agents]  # type: ignore[arg-type]
    for hid in gate.missing_hypotheses:
        statement = state["hypotheses"][hid].statement
        for agent in ("observability", "change_history"):
            requests.append(EvidenceRequest(agent=agent, goal=f"Find evidence for or against {hid}: {statement}",
                                            hypothesis_ids=[hid]))
    return _tasks_from_requests(state, requests)


def _next_round(state: InvestigationState, tasks: list[Task]) -> Decision:
    budget = state["budget"].model_copy(update={"rounds": state["budget"].rounds + 1})
    return _dispatch(state, tasks, "gather", {"budget": budget})


def decide(state: InvestigationState, now: float | None = None) -> Decision:
    """Pick the next step after outcomes for the current phase have been applied."""
    phase = state["phase"]
    limit = check_limits(state, now)

    if phase == "gather":
        if limit:
            return _report(state, limit)
        ready = ready_tasks(state)
        if ready:
            return _dispatch(state, ready, "gather")
        return _dispatch(state, [_single(state, "reasoning", "Rank hypotheses against the evidence.")], "reason")

    if phase == "reason":
        if not any(h.rank is not None for h in state["hypotheses"].values()):
            return _report(state, limit or "no_progress")
        if limit:
            return _report(state, limit)
        return _dispatch(state, [_single(state, "critic", "Challenge the ranked hypotheses against the evidence.")], "critique")

    if phase == "critique":
        if limit:
            return _report(state, limit)
        verdict = _latest_critic_verdict(state)
        loops_left = state["budget"].critic_loops < state["config"].limits.max_critic_loops

        if verdict is not None and verdict.verdict == "needs_human":
            if state["config"].mode == "interactive":
                return Decision(action="human", update={"phase": "human"})
            return _report(state, "needs_human")

        if verdict is not None and verdict.verdict == "need_more_evidence":
            if not loops_left:
                return _report(state, "max_critic_loops")
            new_tasks = _tasks_from_requests(state, verdict.requests)
            if new_tasks:
                return _next_round(state, new_tasks)

        gate = evidence_gate(state)
        if gate.passed:
            reason: StopReason = "inconclusive" if verdict is not None and verdict.verdict == "inconclusive" else "accepted"
            return _report(state, reason)
        if loops_left:
            new_tasks = _tasks_for_gate(state, gate)
            if new_tasks:
                return _next_round(state, new_tasks)
        return _report(state, "max_critic_loops" if not loops_left else "no_progress")

    if phase == "human":
        # Human input arrives as new evidence/outcomes; re-rank with it.
        return _dispatch(state, [_single(state, "reasoning", "Re-rank hypotheses including the human input.")], "reason")

    return _report(state, limit or "no_progress")


def _latest_critic_verdict(state: InvestigationState) -> CriticVerdict | None:
    critic_tasks = [t for t in _sorted_tasks(state["tasks"]) if t.agent == "critic"]
    if not critic_tasks or critic_tasks[-1].status == "failed":
        return None
    return state["verdicts"][-1] if state["verdicts"] else None


def step(state: InvestigationState, now: float | None = None) -> Decision:
    """apply_outcomes + decide, merged into one update (what the orchestrator node returns)."""
    applied = apply_outcomes(state)
    decision = decide(_merged(state, applied), now)
    decision.update = {**applied, **decision.update}
    return decision

