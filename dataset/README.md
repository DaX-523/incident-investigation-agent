# Incident dataset

13 synthetic incident scenarios. Each one is split into two files so the agent can never see the answers:

| Path | Read by | Contents |
|---|---|---|
| `scenarios/inc-NNN.yaml` | tool harness (agent sees the report, plus tool results only) | `incident_report`, `environment` (topology, metrics, logs, traces, changes, infra events, alerts, runbooks, external reports, past incidents), `tool_faults` |
| `answer_keys/inc-NNN.yaml` | evaluators only | `metadata` (title, tags, provenance) and `ground_truth` (root cause and aliases, contributing factors, propagation chain, red herrings, key evidence, expected outcome and confidence, ranked corrective actions, `must_not`) |

Scenario ids are neutral (`inc-NNN`) so ids and filenames don't hint at the answer. `tool_faults` are applied by the tool harness and never returned to the agent. The answer-key format follows the four-layer ground truth in Alibaba's RCA Benchmark: fault type, entity, propagation chain, and evidence. `key_evidence[].tool` and `args` map to agentevals' superset trajectory match.

## Tools the environment backs

`get_topology`, `get_alerts`, `query_metrics`, `search_logs`, `get_traces`, `get_changes`, `get_infra_events`, `search_runbooks`, `get_external_report`, `search_past_incidents`

Fault modes: `timeout`, `error` (optionally recovering after N attempts), `empty`, `stale`, `truncated`.

## Coverage

Run `uv run python -m dataset.validate` for the full matrix.

| id | primary trap | tool fault |
|---|---|---|
| inc-001 | control | none |
| inc-002 | misleading: coincidental deploy | none |
| inc-003 | misleading: symptom far from cause | none |
| inc-004 | misleading: reporter's wrong hypothesis | traces empty |
| inc-005 | conflicting: aggregates hide a bad node | none |
| inc-006 | conflicting: clock skew | none |
| inc-007 | conflicting: health checks vs user impact | none |
| inc-008 | tool failure: log search timeout; **expected outcome inconclusive** | logs timeout |
| inc-009 | tool failure: silent stale metrics | metrics stale |
| inc-010 | incomplete data + multiple causes | metrics flaky (recovers on retry) |
| inc-011 | prompt injection in telemetry | none |
| inc-012 | real reconstruction (Buttercut AWS cost anomaly) | none |
| inc-013 | conflicting: vendor report vs own metrics | logs truncated |

inc-012 is a `real_reconstruction`. Its real figures are limited to what `confirmed_facts` lists, and everything else is marked in `synthetic_details`. All other scenarios use fictional organisations and figures.

## Tests

`uv run pytest` checks every pair, verifies category coverage, and enforces that no code outside `dataset/`, `evals/` or `tests/` references answer keys or scenario ids.
