"""Tool catalogue: name -> input/output models, backend method and description."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from investigator.tools import models as m


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    # The list field in the output that empty/truncated faults and quality checks act on.
    result_field: str
    # Whether omitted start/end default to the incident window (vs. unbounded history).
    window_bounded: bool


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("get_topology", "Service dependency map; with a service, its upstream callers and downstream dependencies.",
                 m.GetTopologyInput, m.TopologyOutput, "services", False),
        ToolSpec("get_alerts", "Alerts that fired or resolved in a time range.",
                 m.GetAlertsInput, m.AlertsOutput, "alerts", True),
        ToolSpec("query_metrics", "Time-series metrics for a service (aggregate by default; label breakdowns on request).",
                 m.QueryMetricsInput, m.MetricsOutput, "series", True),
        ToolSpec("search_logs", "Search application/infrastructure logs (Sentry-style errors included).",
                 m.SearchLogsInput, m.LogsOutput, "entries", True),
        ToolSpec("get_traces", "Distributed traces with per-span timing, status and attributes.",
                 m.GetTracesInput, m.TracesOutput, "traces", True),
        ToolSpec("get_changes", "Deployment and change history: deploys, rollbacks, config, feature flags, migrations, secrets, jobs.",
                 m.GetChangesInput, m.ChangesOutput, "changes", False),
        ToolSpec("get_infra_events", "Infrastructure/platform events: Kubernetes, cloud provider, certificates, clocks, clusters.",
                 m.GetInfraEventsInput, m.InfraEventsOutput, "events", False),
        ToolSpec("search_runbooks", "Operational runbooks.",
                 m.SearchRunbooksInput, m.RunbooksOutput, "runbooks", False),
        ToolSpec("get_external_report", "Third-party/vendor reports: billing, vendor status pages, carrier records, analytics, config snapshots.",
                 m.GetExternalReportInput, m.ExternalReportsOutput, "reports", False),
        ToolSpec("search_past_incidents", "Historical incident postmortems from this organisation (may or may not be relevant).",
                 m.SearchPastIncidentsInput, m.PastIncidentsOutput, "incidents", False),
    ]
}
