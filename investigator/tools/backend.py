"""Simulated observability backends: pure queries over a scenario's environment.

Shaped like the systems they stand in for (Prometheus-style metrics, Sentry-style
logs/errors, GitHub-style deploy history), but read from the scenario YAML.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from dataset.schema import Environment, MetricSeries
from investigator.tools.errors import ToolInvalidRequestError, ToolNotFoundError
from investigator.tools.models import (
    AlertOut,
    AlertsOutput,
    ChangeOut,
    ChangesOutput,
    ExternalReportOut,
    ExternalReportsOutput,
    GetAlertsInput,
    GetChangesInput,
    GetExternalReportInput,
    GetInfraEventsInput,
    GetTopologyInput,
    GetTracesInput,
    InfraEventOut,
    InfraEventsOutput,
    LogOut,
    LogsOutput,
    MetricsOutput,
    PastIncidentOut,
    PastIncidentsOutput,
    Point,
    QueryMetricsInput,
    RunbookOut,
    RunbooksOutput,
    SearchLogsInput,
    SearchPastIncidentsInput,
    SearchRunbooksInput,
    SeriesOut,
    SeriesRef,
    ServiceNode,
    TopologyOutput,
    TraceOut,
    TracesOutput,
)

_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
_AGGREGATE_LABELS = ({}, {"scope": "global"})


def _terms(query: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_.\-/]+", query.lower()) if len(t) >= 3}


def _keyword_rank(items, text_of, query: str | None):
    if not query:
        return list(items)
    terms = _terms(query)
    if not terms:
        return list(items)
    scored = []
    for item in items:
        text = text_of(item).lower()
        score = sum(1 for t in terms if t in text)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: -pair[0])
    return [item for _, item in scored]


def _compile(query: str) -> re.Pattern[str]:
    try:
        return re.compile(query, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(query), re.IGNORECASE)


class ScenarioBackend:
    def __init__(self, env: Environment):
        self.env = env
        self._services = {s.name: s for s in env.topology.services}

    # --- helpers -------------------------------------------------------------

    def window(self, start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
        return start or self.env.window.start, end or self.env.window.end

    def _require_service(self, service: str | None) -> None:
        if service is not None and service not in self._services:
            raise ToolNotFoundError(f"unknown service {service!r}; known services: {sorted(self._services)}")

    # --- tools ---------------------------------------------------------------

    def get_topology(self, inp: GetTopologyInput) -> TopologyOutput:
        nodes = [ServiceNode(**s.model_dump()) for s in self.env.topology.services]
        if inp.service is None:
            return TopologyOutput(services=nodes)
        self._require_service(inp.service)
        callers: dict[str, set[str]] = {name: set() for name in self._services}
        for s in self.env.topology.services:
            for dep in s.depends_on:
                callers[dep].add(s.name)

        def walk(start: str, edges) -> list[str]:
            seen, stack = set(), [start]
            while stack:
                for nxt in edges(stack.pop()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            seen.discard(start)
            return sorted(seen)

        downstream = walk(inp.service, lambda n: self._services[n].depends_on)
        upstream = walk(inp.service, lambda n: callers[n])
        related = {inp.service, *upstream, *downstream}
        return TopologyOutput(
            services=[n for n in nodes if n.name in related], upstream=upstream, downstream=downstream
        )

    def get_alerts(self, inp: GetAlertsInput) -> AlertsOutput:
        start, end = self.window(inp.start, inp.end)
        alerts = [
            AlertOut(**a.model_dump())
            for a in self.env.alerts
            if start <= a.ts <= end and (inp.service is None or inp.service in a.entity)
        ]
        return AlertsOutput(alerts=sorted(alerts, key=lambda a: a.ts))

    def query_metrics(
        self,
        inp: QueryMetricsInput,
        series_source: list[MetricSeries] | None = None,
        clip_to_range: bool = True,
    ) -> MetricsOutput:
        self._require_service(inp.service)
        source = self.env.metrics if series_source is None else series_source
        candidates = [
            s for s in source if s.service == inp.service and (inp.name is None or s.name == inp.name)
        ]
        available = [
            SeriesRef(name=s.name, labels=s.labels)
            for s in self.env.metrics
            if s.service == inp.service and (inp.name is None or s.name == inp.name)
        ]

        if inp.labels:
            selected = [
                s
                for s in candidates
                if all(k in s.labels and (v == "*" or s.labels[k] == v) for k, v in inp.labels.items())
            ]
        else:
            selected = []
            for name in dict.fromkeys(s.name for s in candidates):
                same = [s for s in candidates if s.name == name]
                aggregate = [s for s in same if s.labels in _AGGREGATE_LABELS]
                selected.extend(aggregate or same)

        start, end = self.window(inp.start, inp.end)
        out = []
        for s in selected:
            points = [
                Point(ts=s.start + timedelta(seconds=s.step_seconds * i), value=v)
                for i, v in enumerate(s.values)
            ]
            if clip_to_range:
                points = [p for p in points if start <= p.ts <= end]
            out.append(SeriesOut(service=s.service, name=s.name, unit=s.unit, labels=s.labels, points=points))
        return MetricsOutput(series=out, available_series=available)

    def search_logs(self, inp: SearchLogsInput) -> LogsOutput:
        self._require_service(inp.service)
        start, end = self.window(inp.start, inp.end)
        pattern = _compile(inp.query) if inp.query else None
        min_level = _LEVELS.index(inp.level) if inp.level else 0
        hits = []
        for e in sorted(self.env.logs, key=lambda e: e.ts):
            if not (start <= e.ts <= end):
                continue
            if inp.service and e.service != inp.service:
                continue
            if _LEVELS.index(e.level) < min_level:
                continue
            if pattern and not pattern.search(f"{e.message} {json.dumps(e.attrs, default=str)}"):
                continue
            hits.append(LogOut(**e.model_dump()))
        return LogsOutput(entries=hits[: inp.limit])

    def get_traces(self, inp: GetTracesInput) -> TracesOutput:
        self._require_service(inp.service)
        start, end = self.window(inp.start, inp.end)
        hits = [
            TraceOut(**t.model_dump())
            for t in sorted(self.env.traces, key=lambda t: t.start)
            if start <= t.start <= end
            and (inp.status is None or t.status == inp.status)
            and (inp.service is None or any(sp.service == inp.service for sp in t.spans))
        ]
        return TracesOutput(traces=hits[: inp.limit])

    def get_changes(self, inp: GetChangesInput) -> ChangesOutput:
        end = inp.end or self.env.window.end
        changes = [
            ChangeOut(**c.model_dump())
            for c in self.env.changes
            if (inp.start is None or c.ts >= inp.start)
            and c.ts <= end
            and (inp.service is None or c.service == inp.service)
            and (inp.kind is None or c.kind == inp.kind)
        ]
        return ChangesOutput(changes=sorted(changes, key=lambda c: c.ts))

    def get_infra_events(self, inp: GetInfraEventsInput) -> InfraEventsOutput:
        end = inp.end or self.env.window.end
        events = [
            InfraEventOut(**e.model_dump())
            for e in self.env.infra_events
            if (inp.start is None or e.ts >= inp.start)
            and e.ts <= end
            and (inp.entity is None or inp.entity.lower() in e.entity.lower())
            and (inp.source is None or inp.source.lower() == e.source.lower())
        ]
        return InfraEventsOutput(events=sorted(events, key=lambda e: e.ts))

    def search_runbooks(self, inp: SearchRunbooksInput) -> RunbooksOutput:
        ranked = _keyword_rank(self.env.runbooks, lambda r: f"{r.title} {r.body}", inp.query)
        return RunbooksOutput(runbooks=[RunbookOut(**r.model_dump()) for r in ranked])

    def get_external_report(self, inp: GetExternalReportInput) -> ExternalReportsOutput:
        reports = self.env.external_reports
        if inp.name is None:
            return ExternalReportsOutput(
                reports=[ExternalReportOut(name=r.name, kind=r.kind, generated_at=r.generated_at) for r in reports]
            )
        for r in reports:
            if r.name.lower() == inp.name.lower():
                return ExternalReportsOutput(reports=[ExternalReportOut(**r.model_dump())])
        raise ToolNotFoundError(f"no report named {inp.name!r}; available: {[r.name for r in reports]}")

    def search_past_incidents(self, inp: SearchPastIncidentsInput) -> PastIncidentsOutput:
        ranked = _keyword_rank(
            self.env.past_incidents, lambda p: f"{p.title} {p.summary} {p.root_cause}", inp.query
        )
        return PastIncidentsOutput(incidents=[PastIncidentOut(**p.model_dump()) for p in ranked])


def validate_range(start: datetime | None, end: datetime | None) -> None:
    if start and end and start > end:
        raise ToolInvalidRequestError(f"start {start.isoformat()} is after end {end.isoformat()}")
