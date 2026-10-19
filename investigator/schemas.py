"""Shared investigation state and the messages exchanged between workers and the orchestrator.

Workers (gatherers, reasoning, critic) never write investigation state. Each returns
one `Outcome`, appended to the `outcomes` mailbox; the orchestrator is the only
writer of every other key and applies outcomes in a deterministic order.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

GathererName = Literal["observability", "dependency", "change_history"]
AgentName = Literal["observability", "dependency", "change_history", "reasoning", "critic"]
GATHERERS: tuple[GathererName, ...] = ("observability", "dependency", "change_history")

Confidence = Literal["high", "medium", "low"]
Stance = Literal["supports", "contradicts", "neutral"]
Phase = Literal["triage", "gather", "reason", "critique", "human", "report", "done"]
StopReason = Literal[
    "accepted",
    "inconclusive",
    "needs_human",
    "max_critic_loops",
    "max_tool_calls",
    "max_tokens",
    "max_wall_clock",
    "no_progress",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- configuration -----------------------------------------------------------------


class BudgetLimits(_Model):
    max_critic_loops: int = 3
    max_tool_calls: int = 60
    max_tokens: int = 400_000
    max_wall_clock_s: float = 600.0
    max_tasks_per_round: int = 6
    # Hypotheses (by rank) that must have evidence before the report is allowed.
    gate_top_k: int = 3


class InvestigationConfig(_Model):
    mode: Literal["interactive", "eval"] = "eval"
    limits: BudgetLimits = BudgetLimits()


class Usage(_Model):
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            tool_calls=self.tool_calls + other.tool_calls,
            latency_ms=round(self.latency_ms + other.latency_ms, 2),
        )

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Budget(_Model):
    started_at: float = Field(description="Epoch seconds when the investigation started.")
    usage: Usage = Usage()
    critic_loops: int = 0
    rounds: int = 0


# --- tasks ------------------------------------------------------------------------


class Task(_Model):
    task_id: str
    agent: AgentName
    goal: str
    hypothesis_ids: list[str] = []
    services: list[str] = []
    depends_on: list[str] = []
    status: Literal["pending", "dispatched", "done", "partial", "failed", "skipped"] = "pending"
    round: int = 0


class TaskSpec(_Model):
    """A task proposed by the LLM planner; the orchestrator assigns the real id."""

    key: str = Field(description="Planner-local key, used only to express depends_on between proposed tasks.")
    agent: GathererName
    goal: str
    hypothesis_keys: list[str] = []
    services: list[str] = []
    depends_on: list[str] = Field([], description="Keys of other proposed tasks this one must wait for.")


class HypothesisSpec(_Model):
    key: str
    statement: str
    origin: Literal["reporter", "triage"] = "triage"


class TriagePlan(_Model):
    """Output of the orchestrator's LLM planner for a new incident."""

    summary: str
    affected_services: list[str] = []
    window_start: datetime | None = None
    window_end: datetime | None = None
    hypotheses: list[HypothesisSpec]
    tasks: list[TaskSpec]


# --- evidence ---------------------------------------------------------------------


class HypothesisLink(_Model):
    hypothesis_id: str
    stance: Stance


class ToolCallRecord(_Model):
    """What a worker reports about each tool call it made (copied from ToolResult)."""

    call_id: str
    tool: str
    args: dict[str, Any] = {}
    ok: bool
    error_kind: str | None = None
    error_message: str | None = None
    quality_notes: list[str] = []


class EvidenceDraft(_Model):
    """An observation a gatherer makes, grounded in one or more of its own tool calls."""

    summary: str = Field(description="What was observed, stated as fact from the tool output.")
    source_call_ids: list[str] = Field(min_length=1)
    services: list[str] = []
    links: list[HypothesisLink] = []
    observed_at: datetime | None = None


class Evidence(_Model):
    id: str
    summary: str
    source_call_ids: list[str]
    tools: list[str]
    agent: AgentName
    task_id: str
    round: int
    services: list[str] = []
    links: list[HypothesisLink] = []
    observed_at: datetime | None = None
    quality_notes: list[str] = []


class Gap(_Model):
    kind: Literal["tool_failure", "data_quality", "worker_failure", "not_checked", "budget"]
    description: str
    agent: AgentName | None = None
    task_id: str | None = None
    call_id: str | None = None


# --- hypotheses / reasoning / critic ------------------------------------------------


class Hypothesis(_Model):
    id: str
    statement: str
    origin: Literal["reporter", "triage", "reasoning", "critic"]
    status: Literal["open", "supported", "refuted", "inconclusive"] = "open"
    rank: int | None = None
    confidence: Confidence | None = None
    supporting: list[str] = []
    contradicting: list[str] = []
    assumptions: list[str] = []
    rationale: str | None = None


class HypothesisAssessment(_Model):
    """Reasoning's view of one hypothesis. `hypothesis_id=None` proposes a new hypothesis."""

    hypothesis_id: str | None = None
    statement: str
    rank: int
    status: Literal["open", "supported", "refuted", "inconclusive"]
    confidence: Confidence
    supporting: list[str] = Field([], description="Evidence ids")
    contradicting: list[str] = Field([], description="Evidence ids")
    assumptions: list[str] = []
    rationale: str


class EvidenceRequest(_Model):
    agent: GathererName
    goal: str
    hypothesis_ids: list[str] = []
    services: list[str] = []


class CriticVerdict(_Model):
    verdict: Literal["accept", "need_more_evidence", "inconclusive", "needs_human"]
    issues: list[str] = []
    requests: list[EvidenceRequest] = []
    questions_for_human: list[str] = []


# --- outcome (the only thing workers produce) -----------------------------------------


class Outcome(_Model):
    task_id: str
    agent: AgentName
    status: Literal["ok", "partial", "failed"]
    tool_calls: list[ToolCallRecord] = []
    evidence: list[EvidenceDraft] = []
    gaps: list[str] = []
    errors: list[str] = []
    assessments: list[HypothesisAssessment] | None = None
    verdict: CriticVerdict | None = None
    usage: Usage = Usage()

    @model_validator(mode="after")
    def _payload_matches_agent(self) -> Outcome:
        if self.status == "failed":
            return self
        if self.agent == "reasoning" and self.assessments is None:
            raise ValueError("reasoning outcome needs assessments")
        if self.agent == "critic" and self.verdict is None:
            raise ValueError("critic outcome needs a verdict")
        return self


def failed_outcome(task: Task, error: BaseException | str, usage: Usage | None = None) -> Outcome:
    """What a worker wrapper returns when the worker itself crashes."""
    message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return Outcome(task_id=task.task_id, agent=task.agent, status="failed", errors=[message], usage=usage or Usage())


# --- graph state --------------------------------------------------------------------


class InvestigationState(TypedDict, total=False):
    # Inputs (set once).
    incident: dict[str, Any]
    config: InvestigationConfig
    # Worker mailbox: the ONLY key workers write. List-append reducer is safe under parallel writes.
    outcomes: Annotated[list[Outcome], operator.add]
    # Everything below is written only by the orchestrator.
    outcomes_applied: int
    triage: TriagePlan | None
    tasks: dict[str, Task]
    evidence: list[Evidence]
    hypotheses: dict[str, Hypothesis]
    verdicts: list[CriticVerdict]
    gaps: list[Gap]
    errors: list[str]
    budget: Budget
    phase: Phase
    stop_reason: StopReason | None
    report: Any
