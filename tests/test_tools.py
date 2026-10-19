import json
import random
import threading
import time

import pytest

from dataset.schema import TOOLS
from investigator.scenario import load_scenario, scenario_ids
from investigator.tools import TOOL_SPECS, RetrySettings, ToolHarness

FAST = RetrySettings(max_retries=2, timeout_s=2.0, backoff_base_s=0.2, backoff_cap_s=2.0)


class SleepRecorder:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def make(scenario_id, settings=FAST):
    sleep = SleepRecorder()
    harness = ToolHarness(load_scenario(scenario_id), settings=settings, sleep=sleep, rng=random.Random(7))
    return harness, sleep


def test_registry_matches_dataset_toolset():
    assert set(TOOL_SPECS) == set(TOOLS)


@pytest.mark.parametrize("scenario_id", scenario_ids())
def test_every_tool_runs_on_every_scenario(scenario_id):
    harness, _ = make(scenario_id)
    with harness:
        for tool in TOOL_SPECS:
            args = {"service": harness.scenario.environment.topology.services[0].name} if tool == "query_metrics" else {}
            result = harness.call(tool, args)
            assert result.ok or result.error.kind in {"timeout", "unavailable"}, (tool, result.error)
        assert len(harness.calls) == len(TOOL_SPECS)


@pytest.mark.parametrize("scenario_id", scenario_ids())
def test_fault_notes_never_reach_the_agent(scenario_id):
    harness, _ = make(scenario_id)
    notes = [f.note for f in harness.scenario.tool_faults]
    with harness:
        for tool in TOOL_SPECS:
            for svc in harness.scenario.environment.topology.services:
                args = {"service": svc.name} if tool in {"query_metrics", "search_logs", "get_traces", "get_topology"} else {}
                payload = json.dumps(harness.call(tool, args).model_dump(mode="json"))
                assert not any(note in payload for note in notes)


# --- fault modes ---------------------------------------------------------------


def test_timeout_fault_retries_with_backoff_then_fails():
    harness, sleep = make("inc-008")
    with harness:
        result = harness.call("search_logs", {"service": "orders-worker"})
    assert not result.ok
    assert result.error.kind == "timeout" and result.error.retryable
    assert result.attempts == FAST.max_retries + 1
    assert len(sleep.delays) == FAST.max_retries
    # equal jitter: attempt n waits within [d/2, d] where d = base * 2**n
    for n, delay in enumerate(sleep.delays):
        d = min(FAST.backoff_cap_s, FAST.backoff_base_s * 2**n)
        assert d / 2 <= delay <= d


def test_error_fault_recovers_on_retry():
    harness, sleep = make("inc-010")
    with harness:
        first = harness.call("query_metrics", {"service": "orders-db"})
        second = harness.call("query_metrics", {"service": "orders-db"})
    assert first.ok and first.attempts == 2 and len(sleep.delays) == 1
    assert second.ok and second.attempts == 1


def test_empty_fault_returns_ok_with_empty_flag():
    harness, _ = make("inc-004")
    with harness:
        result = harness.call("get_traces", {})
    assert result.ok and result.data["traces"] == []
    assert result.quality.empty


def test_stale_fault_is_flagged_by_range_check():
    harness, _ = make("inc-009")
    with harness:
        stale = harness.call("query_metrics", {"service": "search-svc"})
        fresh = harness.call("query_metrics", {"service": "es-products"})
    assert stale.ok
    assert stale.quality.out_of_requested_range
    window_start = harness.scenario.environment.window.start
    assert stale.quality.returned_range.end < window_start
    assert fresh.ok and not fresh.quality.out_of_requested_range


def test_truncated_fault_applies_only_to_matching_args():
    harness, _ = make("inc-013")
    with harness:
        truncated = harness.call("search_logs", {"service": "dialer"})
        full = harness.call("search_logs", {})
    assert truncated.ok and len(truncated.data["entries"]) == 2
    assert len(full.data["entries"]) > 2


def test_no_fault_scenarios_are_untouched():
    harness, sleep = make("inc-001")
    with harness:
        result = harness.call("search_logs", {"service": "checkout-svc", "level": "ERROR"})
    assert result.ok and result.attempts == 1 and not sleep.delays
    assert all(e["level"] in {"ERROR", "FATAL"} for e in result.data["entries"])


# --- non-retryable and real failures -----------------------------------------------


def test_invalid_arguments_are_not_retried():
    harness, sleep = make("inc-001")
    with harness:
        bad_level = harness.call("search_logs", {"level": "LOUD"})
        bad_range = harness.call("search_logs", {"start": "2026-09-14T15:00:00Z", "end": "2026-09-14T14:00:00Z"})
        unknown_tool = harness.call("drop_tables", {})
    for r in (bad_level, bad_range, unknown_tool):
        assert not r.ok and r.error.kind == "invalid_request" and not r.error.retryable and r.attempts == 0
    assert not sleep.delays


def test_unknown_service_is_not_found_without_retry():
    harness, sleep = make("inc-001")
    with harness:
        result = harness.call("query_metrics", {"service": "nope-svc"})
    assert result.error.kind == "not_found" and result.attempts == 1 and not sleep.delays
    assert "checkout-svc" in result.error.message


def test_slow_backend_hits_real_timeout_and_is_retried(monkeypatch):
    settings = RetrySettings(max_retries=1, timeout_s=0.05, backoff_base_s=0.01, backoff_cap_s=0.01)
    harness, sleep = make("inc-001", settings)
    monkeypatch.setattr(harness._backend, "get_alerts", lambda inp: time.sleep(0.3))
    with harness:
        result = harness.call("get_alerts", {})
    assert result.error.kind == "timeout" and result.attempts == 2 and len(sleep.delays) == 1


def test_backend_exception_becomes_internal_error(monkeypatch):
    harness, sleep = make("inc-001")

    def boom(inp):
        raise KeyError("bug")

    monkeypatch.setattr(harness._backend, "get_alerts", boom)
    with harness:
        result = harness.call("get_alerts", {})
    assert result.error.kind == "internal" and result.attempts == 1 and not sleep.delays


# --- tool semantics ------------------------------------------------------------------


def test_metrics_default_to_aggregate_and_expose_breakdowns():
    harness, _ = make("inc-002")
    with harness:
        default = harness.call("query_metrics", {"service": "checkout-svc", "name": "http_5xx_rate"})
        by_version = harness.call("query_metrics", {"service": "checkout-svc", "name": "http_5xx_rate", "labels": {"version": "*"}})
    assert [s["labels"] for s in default.data["series"]] == [{}]
    assert {"version": "v3.8.1"} in [ref["labels"] for ref in default.data["available_series"]]
    assert sorted(s["labels"]["version"] for s in by_version.data["series"]) == ["v3.8.0", "v3.8.1"]


def test_metric_gaps_are_flagged():
    harness, _ = make("inc-010")
    with harness:
        result = harness.call("query_metrics", {"service": "api", "name": "requests_per_sec"})
    assert result.quality.null_points == 5


def test_time_range_filters_points():
    harness, _ = make("inc-001")
    with harness:
        result = harness.call("query_metrics", {"service": "checkout-svc", "name": "http_5xx_rate",
                                                "start": "2026-09-14T14:00:00Z", "end": "2026-09-14T14:10:00Z"})
    assert len(result.data["series"][0]["points"]) == 3


def test_topology_blast_radius():
    harness, _ = make("inc-003")
    with harness:
        result = harness.call("get_topology", {"service": "catalog-svc"})
    assert result.data["upstream"] == ["api-gateway"]
    assert result.data["downstream"] == ["catalog-db", "redis-cache"]


def test_changes_include_history_before_the_window():
    harness, _ = make("inc-002")
    with harness:
        result = harness.call("get_changes", {})
    assert any(c["service"] == "cert-manager" for c in result.data["changes"])


def test_external_report_listing_and_lookup():
    harness, _ = make("inc-004")
    with harness:
        listing = harness.call("get_external_report", {})
        report = harness.call("get_external_report", {"name": "cloudflare_status"})
        missing = harness.call("get_external_report", {"name": "nope"})
    assert all(r["body"] is None for r in listing.data["reports"])
    assert "GRU" in report.data["reports"][0]["body"]
    assert missing.error.kind == "not_found"


def test_keyword_search_ranks_past_incidents_and_runbooks():
    harness, _ = make("inc-004")
    with harness:
        hits = harness.call("search_past_incidents", {"query": "auth-svc pool connection"})
        none = harness.call("search_past_incidents", {"query": "kubernetes etcd quorum"})
    assert hits.data["incidents"][0]["title"] == "auth-svc slow logins"
    assert none.ok and none.quality.empty


def test_log_limit_flagged():
    harness, _ = make("inc-001")
    with harness:
        result = harness.call("search_logs", {"limit": 2})
    assert result.quality.at_limit and len(result.data["entries"]) == 2


# --- concurrency -----------------------------------------------------------------------


def test_parallel_calls_get_unique_ids_and_fault_fires_once():
    harness, _ = make("inc-010")
    results = []
    lock = threading.Lock()

    def worker():
        r = harness.call("query_metrics", {"service": "orders-db"})
        with lock:
            results.append(r)

    with harness:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert all(r.ok for r in results)
    assert sum(r.attempts for r in results) == 9  # exactly one injected failure across all callers
    assert len({r.call_id for r in harness.calls}) == 8
