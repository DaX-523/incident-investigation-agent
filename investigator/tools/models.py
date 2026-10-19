"""Structured inputs and outputs for every investigation tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _as_utc(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class TimeRange(_Model):
    start: datetime | None = None
    end: datetime | None = None


class _TimeBounded(_Model):
    start: datetime | None = Field(None, description="Range start (UTC ISO-8601). Defaults to the incident window start.")
    end: datetime | None = Field(None, description="Range end (UTC ISO-8601). Defaults to the incident window end.")

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return _as_utc(v)


# --- inputs ------------------------------------------------------------------


class GetTopologyInput(_Model):
    service: str | None = Field(None, description="If set, return this service with its upstream callers and downstream dependencies (blast radius). Otherwise the full service map.")


class GetAlertsInput(_TimeBounded):
    service: str | None = Field(None, description="Only alerts for this entity/service.")


class QueryMetricsInput(_TimeBounded):
    service: str = Field(description="Service or entity name from the topology.")
    name: str | None = Field(None, description="Metric name. Omit to get every metric for the service.")
    labels: dict[str, str] | None = Field(
        None,
        description="Label filter, e.g. {'az': 'eu-west-1c'}; use '*' as a value to get every series with that label. "
        "Omit to get aggregate series; the response lists the available label breakdowns.",
    )


class SearchLogsInput(_TimeBounded):
    query: str | None = Field(None, description="Case-insensitive substring or regex matched against the message and attributes.")
    service: str | None = None
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"] | None = Field(None, description="Minimum severity.")
    limit: int = Field(100, ge=1, le=500)


class GetTracesInput(_TimeBounded):
    service: str | None = Field(None, description="Only traces with a span in this service.")
    status: Literal["ok", "error"] | None = None
    limit: int = Field(20, ge=1, le=200)


class GetChangesInput(_Model):
    start: datetime | None = Field(None, description="Defaults to unbounded (full change history).")
    end: datetime | None = Field(None, description="Defaults to the incident window end.")
    service: str | None = None
    kind: str | None = Field(None, description="deploy, rollback, config, feature_flag, infra, secret, migration, scheduled_job")

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return _as_utc(v)


class GetInfraEventsInput(_Model):
    start: datetime | None = Field(None, description="Defaults to unbounded.")
    end: datetime | None = Field(None, description="Defaults to the incident window end.")
    entity: str | None = Field(None, description="Substring match on the entity, e.g. a pod or node name.")
    source: str | None = Field(None, description="e.g. kubernetes, cert-manager, aws-ec2")

    @field_validator("start", "end")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        return _as_utc(v)


class SearchRunbooksInput(_Model):
    query: str | None = Field(None, description="Keywords. Omit to list all runbooks.")


class GetExternalReportInput(_Model):
    name: str | None = Field(None, description="Report name. Omit to list available reports (billing, vendor status, CDRs, analytics, config snapshots).")


class SearchPastIncidentsInput(_Model):
    query: str | None = Field(None, description="Keywords matched against title, summary and root cause. Omit to list all.")


# --- outputs -----------------------------------------------------------------


class ServiceNode(_Model):
    name: str
    kind: str
    depends_on: list[str] = []
    notes: str | None = None


class TopologyOutput(_Model):
    services: list[ServiceNode]
    upstream: list[str] = Field([], description="Services that (transitively) call the requested service.")
    downstream: list[str] = Field([], description="Services the requested service (transitively) depends on.")


class AlertOut(_Model):
    ts: datetime
    name: str
    severity: str
    entity: str
    state: str
    summary: str


class AlertsOutput(_Model):
    alerts: list[AlertOut]


class Point(_Model):
    ts: datetime
    value: float | None


class SeriesOut(_Model):
    service: str
    name: str
    unit: str
    labels: dict[str, str] = {}
    points: list[Point]


class SeriesRef(_Model):
    name: str
    labels: dict[str, str] = {}


class MetricsOutput(_Model):
    series: list[SeriesOut]
    available_series: list[SeriesRef] = Field([], description="All series that exist for this service (for label breakdowns).")


class LogOut(_Model):
    ts: datetime
    service: str
    level: str
    message: str
    attrs: dict[str, Any] = {}


class LogsOutput(_Model):
    entries: list[LogOut]


class SpanOut(_Model):
    service: str
    operation: str
    duration_ms: float
    status: str
    attrs: dict[str, Any] = {}


class TraceOut(_Model):
    trace_id: str
    start: datetime
    duration_ms: float
    status: str
    spans: list[SpanOut]


class TracesOutput(_Model):
    traces: list[TraceOut]


class ChangeOut(_Model):
    ts: datetime
    kind: str
    service: str
    summary: str
    author: str
    details: str | None = None


class ChangesOutput(_Model):
    changes: list[ChangeOut]


class InfraEventOut(_Model):
    ts: datetime
    source: str
    entity: str
    reason: str
    message: str


class InfraEventsOutput(_Model):
    events: list[InfraEventOut]


class RunbookOut(_Model):
    id: str
    title: str
    body: str


class RunbooksOutput(_Model):
    runbooks: list[RunbookOut]


class ExternalReportOut(_Model):
    name: str
    kind: str
    generated_at: datetime
    body: str | None = Field(None, description="Omitted in the listing view.")


class ExternalReportsOutput(_Model):
    reports: list[ExternalReportOut]


class PastIncidentOut(_Model):
    id: str
    date: datetime
    title: str
    summary: str
    root_cause: str
    resolution: str


class PastIncidentsOutput(_Model):
    incidents: list[PastIncidentOut]


# --- harness envelope ----------------------------------------------------------


ErrorKind = Literal["timeout", "unavailable", "invalid_request", "not_found", "internal"]


class ToolError(_Model):
    kind: ErrorKind
    message: str
    retryable: bool


class DataQuality(_Model):
    """Generic checks the harness runs on every successful result."""

    requested_range: TimeRange | None = None
    returned_range: TimeRange | None = None
    result_count: int = 0
    empty: bool = False
    out_of_requested_range: bool = Field(False, description="Returned timestamps fall outside the requested range (possible stale/cached data).")
    null_points: int = Field(0, description="Metric points with no value (scrape/retention gaps).")
    at_limit: bool = Field(False, description="Result count equals the requested limit; more results may exist.")
    notes: list[str] = []


class ToolResult(_Model):
    call_id: str
    tool: str
    args: dict[str, Any]
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    quality: DataQuality | None = None
    attempts: int
    latency_ms: float
