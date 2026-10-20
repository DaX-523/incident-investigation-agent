# Multi-agent vs single-agent baseline (budget, openai)

Generated 2026-10-20T14:08:19+0530. Scenarios: 3. Scored against `dataset/answer_keys/`: deterministic checks in `evals/scoring.py`; root cause, red herrings, must-not and factors graded by LLM judge `openai:gpt-5.4-mini` (`evals/judge.py`).

Some runs were saved without tool data, so their key evidence is checked by tool and arguments only (`evidence_verified: false` in results.json).

- **multi_agent** models: triage=gpt-5.4/medium, observability=gpt-5.4/medium, change_history=gpt-5.4/medium, dependency=gpt-5.4/low, reasoning=o3/high, critic=o3/high, reporter=gpt-5.4/medium
- **baseline** models: baseline=o3/high

## Summary

| metric | multi_agent | baseline |
|---|---|---|
| runs | 3 | 3 |
| root_cause_accuracy | 0.67 | 0.67 |
| root_cause_accuracy_lexical | 1.00 | 0.67 |
| root_cause_accuracy_incl_review | 1.00 | 0.67 |
| needs_review | 1 | 0 |
| errors | 0 | 0 |
| outcome_correct | 1.00 | 0.67 |
| claims_grounded | 1.00 | 0.85 |
| citations_valid | 1.00 | 1.00 |
| key_evidence_retrieved | 1.00 | 1.00 |
| key_evidence_cited | 1.00 | 0.61 |
| factors_mentioned | 0.67 | 0.50 |
| fell_for_red_herring | 1 | 0 |
| must_not_violations | 1 | 0 |
| security_flagged | 0/0 | 0/0 |
| tool_calls_mean | 52.3 | 32.3 |
| llm_calls_mean | 31.0 | 33.3 |
| latency_s_mean | 290.0 | 251.0 |
| latency_s_p50 | 285.5 | 280.8 |
| input_tokens_total | 593352 | 987845 |
| output_tokens_total | 108913 | 50912 |
| tokens_mean | 234088 | 346252 |
| cost_usd_total | 2.29 | 1.00 |
| cost_usd_mean | 0.76 | 0.33 |

## Root-cause accuracy by scenario type

| type | multi_agent | baseline |
|---|---|---|
| misleading_evidence | 1/1 | 1/1 |
| premature_closure | 0/1 | 0/1 |
| tool_failure | 1/1 | 1/1 |

## Per scenario

| scenario | type | expected | system | verdict | outcome/conf | key ev. cited | grounded | tools | latency s | tokens | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inc-003 | misleading_evidence | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 40 | 253.4 | 169807 | 0.6160 |
| inc-003 | misleading_evidence | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 21 | 154.0 | 170137 | 0.1890 |
| inc-008 | tool_failure | inconclusive/low | multi_agent | ✅ correct | inconclusive/low | 1.00 | 1.00 | 60 | 331.2 | 260783 | 0.8216 |
| inc-008 | tool_failure | inconclusive/low | baseline | ✅ correct | inconclusive/low | 0.50 | 0.71 | 38 | 280.8 | 433837 | 0.3800 |
| inc-016 | premature_closure | conclusive/high | multi_agent | 🟡 review | conclusive/medium | 1.00 | 1.00 | 57 | 285.5 | 271675 | 0.8494 |
| inc-016 | premature_closure | conclusive/high | baseline | ❌ wrong | inconclusive/low | 0.33 | 0.83 | 38 | 318.4 | 434783 | 0.4303 |

## Not correct, or worth a look

- **inc-016/baseline** (wrong; lexical wrong, alias match 0.0): (no root cause)
  - judge: The report identifies the same failure mechanism—tracking-db IOPS saturation causing slow queries and connection exhaustion—but it does not name the originating backfill job/component that caused the saturation, so it is only a partial match to the key.
- **inc-016/multi_agent** (review; lexical correct, alias match 1.0): A mis-scheduled data-jobs back-fill (job/backfill-carrier-events) started at 08:49 with 16 parallel pods, wrote heavily to tracking-db, saturated write IOPS, drove query latency and lock waits up from 08:50, which made tracking-api hold DB connections longer; when FinOps simultaneously lowered tracking-api’s pool to 40 connections, the pool hit 100 % utilisation and partners began receiving 5xx errors.
  - judge: The report identifies the correct originating job, backfill-carrier-events, and the same saturation mechanism on tracking-db, but it makes the 08:55 tracking-api pool reduction part of the primary cause rather than a contributing factor. That means it finds the right cause only in combination with an additional different primary cause.
  - adopted red herring: ['The 08:55 pool size reduction is the root cause, so reverting it fixes the incident.']
  - violated must-not: ['Name the DB pool size reduction as the root cause.']

## Metric definitions

- **root_cause_accuracy**: share of runs whose root cause the judge grades as the answer key's root cause (not merely mentioned as an amplifier or trigger); for expected-inconclusive scenarios, share that reported inconclusive. `review` = judge "partial" (or, without a judge, a partial alias match), decided manually in `reviews.yaml`. **root_cause_accuracy_lexical**: the same by alias word overlap only.
- **fell_for_red_herring / must_not_violations**: runs that adopt a red herring as cause or primary fix / do something the key forbids (judge).
- **claims_grounded**: share of supported/refuted hypotheses and corrective actions citing at least one evidence id that traces to an executed tool call. **citations_valid**: share of all cited ids that do.
- **key_evidence_retrieved / cited**: share of the answer key's required evidence the system fetched (a successful call whose returned data contains the evidence) / cited in its report.
- **factors_mentioned**: share of must-mention contributing factors the report states (judge; lexical word overlap without a judge).
- Tokens include cache reads/writes; cost uses `investigator/llm/pricing.py`. Latency is wall clock per run.
