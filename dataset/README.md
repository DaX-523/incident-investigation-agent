# Incident dataset

16 incident scenarios (15 synthetic, 1 real reconstruction). Each one is split into two files so the agent can never see the answers:

| Path | Read by | Contents |
|---|---|---|
| `scenarios/inc-NNN.yaml` | tool harness (agent sees the report, plus tool results only) | `incident_report`, `environment` (topology, metrics, logs, traces, changes, infra events, alerts, runbooks, external reports, past incidents), `tool_faults` |
| `answer_keys/inc-NNN.yaml` | evaluators only | `metadata` (title, tags, provenance) and `ground_truth` (root cause and aliases, contributing factors, propagation chain, red herrings, key evidence, expected outcome and confidence, ranked corrective actions, `must_not`) |

`key_evidence[].must_contain` lists strings the returned tool data must hold for a call to count as retrieving that evidence, so a filtered query that could not return it doesn't count. `attempt_only` marks evidence that is the attempt itself (trying a source that fails). The validator checks every `must_contain` string occurs in the scenario, and a test checks the key's own arguments can return it.

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
| inc-014 | premature closure: release blamed; decisive evidence in traces + infra events (cache failover) | none |
| inc-015 | premature closure: client upgrade blamed; decisive evidence in traces + infra events (DNS replicas evicted) | none |
| inc-016 | premature closure: real but minor pool cut vs backfill job saturating the DB | none |

The `premature_closure` scenarios target a failure mode measured in our own runs: the single-agent baseline stops once it has a convincing story and often never calls `get_traces` or `get_infra_events`. Each has a plausible explanation in the first sources an investigator checks (metrics, logs, recent changes, the reporter's theory), while the evidence that overturns it is in sources usually checked later. The overturning evidence is reachable with the same tools by either system, and the first sources also contain weaker hints against the decoy.

inc-012 is a `real_reconstruction`. Its real figures are limited to what `confirmed_facts` lists, and everything else is marked in `synthetic_details`. All other scenarios use fictional organisations and figures.

## Tests

`uv run pytest` checks every pair, verifies category coverage, and enforces that no code outside `dataset/`, `evals/` or `tests/` references answer keys or scenario ids.
