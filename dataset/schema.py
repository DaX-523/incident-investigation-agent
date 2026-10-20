"""Schema for incident scenarios and their answer keys.

A scenario (dataset/scenarios) is a self-contained "world": the incident report
the agent receives, the data it can reach only through tools, and faults the
tool harness injects. The answer key (dataset/answer_keys) holds metadata and
ground truth; only evaluators may load it, never agent or tool code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The fixed toolset the agent gets in every scenario. Mock tools read from
# `Environment`; `ToolFault` entries alter what they return.
TOOLS = (
    "get_topology",
    "get_alerts",
    "query_metrics",
    "search_logs",
    "get_traces",
    "get_changes",
    "get_infra_events",
    "search_runbooks",
    "get_external_report",
    "search_past_incidents",
)
ToolName = Literal[
    "get_topology",
    "get_alerts",
    "query_metrics",
    "search_logs",
    "get_traces",
    "get_changes",
    "get_infra_events",
    "search_runbooks",
    "get_external_report",
    "search_past_incidents",
]

Tag = Literal[
    "control",
    "misleading_evidence",
    "conflicting_observability",
    "tool_failure",
    "incomplete_data",
    "multi_cause",
    "prompt_injection",
    "real_incident",
    # A plausible, self-consistent explanation sits in the first sources an investigator checks;
    # the evidence that overturns it is only in sources checked later (traces, infra events, ...).
    "premature_closure",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- metadata & report -------------------------------------------------------


class Metadata(Strict):
    id: str
    title: str
    primary_tag: Tag
    tags: list[Tag]
    difficulty: Literal["L1", "L2", "L3", "L4"]
    provenance: Literal["synthetic", "real_reconstruction"]
    inspired_by: str
    # Real reconstructions only: what is backed by source material vs. invented
    # filler, and what the incident owner still needs to confirm.
    confirmed_facts: list[str] = []
    synthetic_details: list[str] = []
    to_confirm: list[str] = []


class IncidentReport(Strict):
    reported_at: datetime
    reporter: str
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"]
    title: str
    description: str


# --- environment (only reachable through tools) ------------------------------


class Service(Strict):
    name: str
    kind: str
    depends_on: list[str] = []
    notes: str | None = None


class Topology(Strict):
    services: list[Service]


class MetricSeries(Strict):
    service: str
    name: str
    unit: str
    labels: dict[str, str] = {}
    start: datetime
    step_seconds: int
    # None marks a gap (no data scraped / retained).
    values: list[float | None]


class LogEntry(Strict):
    ts: datetime
    service: str
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    message: str
    attrs: dict[str, Any] = {}


class Span(Strict):
    service: str
    operation: str
    duration_ms: float
    status: Literal["ok", "error"]
    attrs: dict[str, Any] = {}


class Trace(Strict):
    trace_id: str
    start: datetime
    duration_ms: float
    status: Literal["ok", "error"]
    spans: list[Span]


class Change(Strict):
    ts: datetime
    kind: Literal["deploy", "rollback", "config", "feature_flag", "infra", "secret", "migration", "scheduled_job"]
    service: str
    summary: str
    author: str
    details: str | None = None


class InfraEvent(Strict):
    ts: datetime
    source: str
    entity: str
    reason: str
    message: str


class Alert(Strict):
    ts: datetime
    name: str
    severity: Literal["critical", "warning", "info"]
    entity: str
    state: Literal["firing", "resolved"]
    summary: str


class Runbook(Strict):
    id: str
    title: str
    body: str


class ExternalReport(Strict):
    name: str
    kind: str
    generated_at: datetime
    body: str


class PastIncident(Strict):
    """A historical postmortem from the same organisation; may or may not be relevant."""

    id: str
    date: datetime
    title: str
    summary: str
    root_cause: str
    resolution: str


class TimeWindow(Strict):
    start: datetime
    end: datetime


class Environment(Strict):
    window: TimeWindow
    topology: Topology
    metrics: list[MetricSeries] = []
    logs: list[LogEntry] = []
    traces: list[Trace] = []
    changes: list[Change] = []
    infra_events: list[InfraEvent] = []
    alerts: list[Alert] = []
    runbooks: list[Runbook] = []
    external_reports: list[ExternalReport] = []
    past_incidents: list[PastIncident] = []


class ToolFault(Strict):
    """Applied by the tool harness. Never returned to the agent (including `note`)."""

    tool: ToolName
    # Argument filters (all must match). Keys: service, name, query_regex, source.
    match: dict[str, str] = {}
    mode: Literal["timeout", "error", "empty", "stale", "truncated"]
    # How many calls fail before the tool starts working; None = never recovers.
    fails_for_attempts: int | None = None
    error_message: str | None = None
    # stale: series returned instead of the real ones (normal-looking, old timestamps).
    substitute_metrics: list[MetricSeries] = []
    # truncated: return only the first N results, with no indication.
    keep_first_n: int | None = None
    note: str


# --- answer key (evaluators only) -------------------------------------------


class RootCause(Strict):
    statement: str
    fault_type: str
    entity: str
    aliases: list[str] = Field(min_length=1)


class ContributingFactor(Strict):
    statement: str
    must_mention: bool = False


class RedHerring(Strict):
    claim: str
    why_wrong: str
    disproved_by: list[str]  # KeyEvidence ids


class KeyEvidence(Strict):
    id: str
    tool: ToolName
    # Partial args a matching tool call must contain (agentevals subset match).
    args: dict[str, Any] = {}
    description: str
    required: bool = True
    # Case-insensitive strings the call's returned data must all contain, so a filtered query that
    # could not return this evidence doesn't count. Each must occur in the scenario environment.
    must_contain: list[str] = []
    # The evidence is the attempt itself (e.g. trying a source that fails); a failed call counts.
    attempt_only: bool = False


class CorrectiveAction(Strict):
    rank: int
    category: Literal["mitigate", "fix", "prevent", "detect"]
    action: str
    rationale: str


class GroundTruth(Strict):
    root_cause: RootCause
    contributing_factors: list[ContributingFactor] = []
    propagation_chain: list[str]
    red_herrings: list[RedHerring] = []
    key_evidence: list[KeyEvidence]
    # inconclusive: the correct report says evidence is insufficient to name a root cause.
    expected_outcome: Literal["conclusive", "inconclusive"] = "conclusive"
    expected_confidence: Literal["high", "medium", "low"]
    missing_data_to_request: list[str] = []
    corrective_actions: list[CorrectiveAction]
    must_not: list[str] = []
    security_findings: list[str] = []


class Scenario(Strict):
    """Everything the harness needs to run the agent. Contains no answers."""

    id: str
    incident_report: IncidentReport
    environment: Environment
    tool_faults: list[ToolFault] = []


class AnswerKey(Strict):
    """Evaluator-only: metadata (titles/tags would hint at the answer) and ground truth."""

    id: str
    metadata: Metadata
    ground_truth: GroundTruth
