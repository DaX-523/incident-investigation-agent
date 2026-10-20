"""Read-only snapshots the orchestrator hands to workers through `Send`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from investigator.orchestrator.manager import ranked_hypotheses
from investigator.schemas import (
    CriticVerdict,
    Evidence,
    Gap,
    Hypothesis,
    InvestigationState,
    StopReason,
    Task,
    ToolCallRecord,
    TriagePlan,
)


class WorkerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident: dict[str, Any]
    triage: TriagePlan | None = None
    hypotheses: list[Hypothesis] = []
    evidence: list[Evidence] = []
    gaps: list[Gap] = []
    tool_calls: list[ToolCallRecord] = []  # every query the team has run so far
    last_verdict: CriticVerdict | None = None
    human_inputs: list[str] = []
    tool_call_allowance: int = 0
    max_worker_attempts: int = 2
    critic_rounds_left: int = 0
    human_available: bool = False
    budget_note: str = ""
    final_call: StopReason | None = None  # set when a limit was hit: decide now, no more gathering


class WorkerInput(TypedDict):
    task: Task
    context: WorkerContext


def _budget_note(state: InvestigationState) -> str:
    limits, budget = state["config"].limits, state["budget"]
    rounds = max(1, budget.rounds)
    return (f"Budget used: {budget.usage.tokens:,}/{limits.max_tokens:,} tokens, {budget.usage.tool_calls}/"
            f"{limits.max_tool_calls} tool calls over {budget.rounds} evidence round(s); one round has cost about "
            f"{budget.usage.tokens // rounds:,} tokens and {budget.usage.tool_calls // rounds} tool calls.")


def build_context(state: InvestigationState, task: Task) -> WorkerContext:
    limits = state["config"].limits
    remaining = max(0, limits.max_tool_calls - state["budget"].usage.tool_calls)
    is_gatherer = task.agent in ("observability", "dependency", "change_history")
    return WorkerContext(
        incident=state["incident"],
        triage=state.get("triage"),
        hypotheses=ranked_hypotheses(state),
        evidence=list(state["evidence"]),
        gaps=list(state["gaps"]),
        tool_calls=[c for o in state.get("outcomes", []) for c in o.tool_calls],
        last_verdict=state["verdicts"][-1] if state.get("verdicts") else None,
        human_inputs=list(state.get("human_inputs", [])),
        tool_call_allowance=min(remaining, limits.max_tool_calls_per_task) if is_gatherer else 0,
        max_worker_attempts=limits.max_worker_attempts,
        critic_rounds_left=0 if state["budget"].limit_hit else max(0, limits.max_critic_loops - state["budget"].critic_loops - 1),
        human_available=state["config"].mode == "interactive",
        budget_note=_budget_note(state),
        final_call=state["budget"].limit_hit,
    )
