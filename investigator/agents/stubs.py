"""Deterministic, LLM-free workers for wiring tests and offline dry runs.

Generic by design: they run a fixed tool sweep per agent and never look at
scenario-specific content beyond what tools return.
"""

from __future__ import annotations

from collections.abc import Sequence

from investigator.orchestrator.context import WorkerContext
from investigator.schemas import (
    GATHERERS,
    CriticVerdict,
    EvidenceDraft,
    HypothesisAssessment,
    HypothesisLink,
    HypothesisSpec,
    Outcome,
    Task,
    TaskSpec,
    TriagePlan,
)
from investigator.tools.toolbox import ToolBox

SWEEPS: dict[str, list[tuple[str, dict]]] = {
    "observability": [("get_alerts", {}), ("search_logs", {"level": "WARN"}), ("get_traces", {"status": "error"})],
    "dependency": [("get_topology", {})],
    "change_history": [("get_changes", {}), ("get_infra_events", {}), ("search_past_incidents", {})],
}


def stub_planner(incident: dict, toolbox: ToolBox) -> TriagePlan:
    toolbox.call("get_topology", {})
    claim = f"{incident.get('title', '')}".strip()
    return TriagePlan(
        summary=f"Investigate: {claim}",
        hypotheses=[
            HypothesisSpec(key="reporter", statement=f"Reporter's account: {claim}", origin="reporter"),
            HypothesisSpec(key="change", statement="A recent change triggered the incident"),
        ],
        tasks=[TaskSpec(key=a, agent=a, goal=f"{a} sweep", hypothesis_keys=["reporter", "change"]) for a in GATHERERS],
    )


def make_stub_gatherer(agent: str):
    def gather(task: Task, context: WorkerContext, toolbox: ToolBox) -> Outcome:
        evidence, gaps = [], []
        for tool, args in SWEEPS[agent]:
            result = toolbox.call(tool, args)
            if not result.ok:
                gaps.append(f"{tool} unavailable: {result.error.kind}")
                continue
            count = result.quality.result_count if result.quality else 0
            evidence.append(EvidenceDraft(
                summary=f"{tool} returned {count} result(s)",
                source_call_ids=[result.call_id],
                links=[HypothesisLink(hypothesis_id=h, stance="neutral") for h in task.hypothesis_ids],
            ))
        status = "ok" if not gaps else ("partial" if evidence else "failed")
        return Outcome(task_id=task.task_id, agent=task.agent, status=status, evidence=evidence, gaps=gaps)

    return gather


def stub_reasoner(task: Task, context: WorkerContext) -> Outcome:
    evidence_ids = [e.id for e in context.evidence]
    assessments = []
    for rank, h in enumerate(sorted(context.hypotheses, key=lambda h: h.id), start=1):
        top = rank == 1 and bool(evidence_ids)
        assessments.append(HypothesisAssessment(
            hypothesis_id=h.id,
            statement=h.statement,
            rank=rank,
            status="supported" if top else ("refuted" if evidence_ids else "open"),
            confidence="medium" if top else "low",
            supporting=evidence_ids[:5] if top else [],
            contradicting=[] if top else evidence_ids[-1:],
            rationale="stub ranking",
        ))
    return Outcome(task_id=task.task_id, agent="reasoning", status="ok", assessments=assessments)


class ScriptedCritic:
    """Returns verdicts in order; repeats the last one once the script runs out."""

    def __init__(self, verdicts: Sequence[CriticVerdict] = (CriticVerdict(verdict="accept"),)):
        self.verdicts = list(verdicts)
        self.calls = 0

    def __call__(self, task: Task, context: WorkerContext) -> Outcome:
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return Outcome(task_id=task.task_id, agent="critic", status="ok", verdict=verdict)


def stub_workers(critic: ScriptedCritic | None = None):
    from investigator.graph import Workers

    return Workers(
        planner=stub_planner,
        gatherers={a: make_stub_gatherer(a) for a in GATHERERS},
        reasoner=stub_reasoner,
        critic=critic or ScriptedCritic(),
    )
