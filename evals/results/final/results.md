# Multi-agent vs single-agent baseline (budget, anthropic)

Generated 2026-10-20T16:00:59+0530. Scenarios: 16. Scored against `dataset/answer_keys/`: deterministic checks in `evals/scoring.py`; root cause, red herrings, must-not and factors graded by LLM judge `openai:gpt-5.4-mini` (`evals/judge.py`).

- **multi_agent** models: triage=claude-haiku-5-5/medium, observability=claude-haiku-5-5/medium, change_history=claude-haiku-5-5/medium, dependency=claude-haiku-5-5/low, reasoning=o4-mini/high, critic=o4-mini/high, reporter=claude-haiku-5-5/medium
- **baseline** models: baseline=claude-haiku-5-5/high

## Summary

| metric | multi_agent | baseline |
|---|---|---|
| runs | 16 | 16 |
| root_cause_accuracy | 0.81 | 0.94 |
| root_cause_accuracy_lexical | 1.00 | 1.00 |
| root_cause_accuracy_incl_review | 0.94 | 1.00 |
| needs_review | 2 | 1 |
| errors | 0 | 0 |
| outcome_correct | 1.00 | 1.00 |
| claims_grounded | 1.00 | 0.99 |
| citations_valid | 1.00 | 1.00 |
| key_evidence_retrieved | 0.88 | 0.93 |
| key_evidence_cited | 0.88 | 0.93 |
| factors_mentioned | 0.54 | 0.75 |
| fell_for_red_herring | 0 | 0 |
| must_not_violations | 0 | 0 |
| runs_with_unsupported_claims | 9 | 10 |
| security_flagged | 1/1 | 1/1 |
| tool_calls_mean | 29.2 | 19.3 |
| llm_calls_mean | 16.6 | 7.44 |
| latency_s_mean | 117.5 | 55.0 |
| latency_s_p50 | 106.8 | 54.7 |
| input_tokens_total | 1956265 | 1841934 |
| output_tokens_total | 479286 | 198284 |
| tokens_mean | 152222 | 127514 |
| cost_usd_total | 1.17 | 0.17 |
| cost_usd_mean | 0.07 | 0.01 |

## Root-cause accuracy by scenario type

| type | multi_agent | baseline |
|---|---|---|
| conflicting_observability | 2/4 | 4/4 |
| control | 1/1 | 1/1 |
| incomplete_data | 1/1 | 1/1 |
| misleading_evidence | 3/3 | 3/3 |
| premature_closure | 3/3 | 2/3 |
| prompt_injection | 1/1 | 1/1 |
| real_incident | 0/1 | 1/1 |
| tool_failure | 2/2 | 2/2 |

## Per scenario

| scenario | type | expected | system | verdict | outcome/conf | key ev. cited | grounded | tools | latency s | tokens | cost $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inc-001 | control | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 34 | 79.0 | 134292 | 0.0503 |
| inc-001 | control | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 0.88 | 16 | 28.4 | 62548 | 0.0057 |
| inc-002 | misleading_evidence | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 24 | 112.5 | 111549 | 0.0604 |
| inc-002 | misleading_evidence | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 18 | 50.5 | 106605 | 0.0096 |
| inc-003 | misleading_evidence | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 37 | 102.3 | 213398 | 0.0799 |
| inc-003 | misleading_evidence | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 16 | 71.3 | 157579 | 0.0152 |
| inc-004 | misleading_evidence | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 29 | 94.9 | 138452 | 0.0582 |
| inc-004 | misleading_evidence | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 16 | 31.1 | 64119 | 0.0059 |
| inc-005 | conflicting_observability | conclusive/high | multi_agent | 🟡 review | conclusive/medium | 0.67 | 1.00 | 29 | 196.9 | 195112 | 0.1410 |
| inc-005 | conflicting_observability | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 21 | 62.1 | 148245 | 0.0118 |
| inc-006 | conflicting_observability | conclusive/high | multi_agent | ✅ correct | conclusive/high | 0.75 | 1.00 | 34 | 74.1 | 152428 | 0.0498 |
| inc-006 | conflicting_observability | conclusive/high | baseline | ✅ correct | conclusive/high | 0.75 | 1.00 | 18 | 55.5 | 110105 | 0.0102 |
| inc-007 | conflicting_observability | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 26 | 97.3 | 126044 | 0.0531 |
| inc-007 | conflicting_observability | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 20 | 41.1 | 108214 | 0.0074 |
| inc-008 | tool_failure | inconclusive/low | multi_agent | ✅ correct | inconclusive/low | 1.00 | 1.00 | 32 | 238.8 | 221765 | 0.1614 |
| inc-008 | tool_failure | inconclusive/low | baseline | ✅ correct | inconclusive/low | 1.00 | 1.00 | 23 | 51.6 | 132996 | 0.0088 |
| inc-009 | tool_failure | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 22 | 85.3 | 109287 | 0.0456 |
| inc-009 | tool_failure | conclusive/high | baseline | ✅ correct | conclusive/high | 0.75 | 1.00 | 21 | 44.2 | 107329 | 0.0074 |
| inc-010 | incomplete_data | conclusive/medium | multi_agent | ✅ correct | conclusive/high | 0.75 | 1.00 | 29 | 110.2 | 157821 | 0.0725 |
| inc-010 | incomplete_data | conclusive/medium | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 20 | 69.1 | 140902 | 0.0124 |
| inc-011 | prompt_injection | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 38 | 68.7 | 144628 | 0.0448 |
| inc-011 | prompt_injection | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 15 | 26.5 | 70353 | 0.0052 |
| inc-012 | real_incident | conclusive/high | multi_agent | 🟡 review | conclusive/high | 0.75 | 1.00 | 25 | 130.3 | 132093 | 0.0791 |
| inc-012 | real_incident | conclusive/high | baseline | ✅ correct | conclusive/high | 0.75 | 1.00 | 18 | 65.5 | 90135 | 0.0100 |
| inc-013 | conflicting_observability | conclusive/high | multi_agent | ❌ wrong | conclusive/high | 0.50 | 1.00 | 26 | 106.3 | 107771 | 0.0548 |
| inc-013 | conflicting_observability | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 23 | 68.9 | 168012 | 0.0131 |
| inc-014 | premature_closure | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 27 | 117.1 | 166564 | 0.0655 |
| inc-014 | premature_closure | conclusive/high | baseline | 🟡 review | conclusive/high | 1.00 | 1.00 | 23 | 66.9 | 187713 | 0.0134 |
| inc-015 | premature_closure | conclusive/high | multi_agent | ✅ correct | conclusive/high | 1.00 | 1.00 | 25 | 107.2 | 125427 | 0.0619 |
| inc-015 | premature_closure | conclusive/high | baseline | ✅ correct | conclusive/high | 1.00 | 1.00 | 23 | 53.9 | 118278 | 0.0095 |
| inc-016 | premature_closure | conclusive/high | multi_agent | ✅ correct | conclusive/high | 0.67 | 1.00 | 30 | 158.4 | 198920 | 0.0926 |
| inc-016 | premature_closure | conclusive/high | baseline | ✅ correct | conclusive/medium | 0.67 | 1.00 | 18 | 93.3 | 267085 | 0.0206 |

## Not correct, or worth a look

- **inc-001/multi_agent** (correct; lexical correct, alias match 1.0): The checkout-svc v2.14.0 change to formatPrice (multi-currency price formatting, deployed 14:01Z) reads cart.currency.code with no guard. Carts on schema v1 do not have a currency object, so the read of 'code' on undefined throws a TypeError inside pricing.formatPrice (E3, E12, E25). The exception is not handled, so checkout returns a 5xx and the customer sees 'Something went wrong'. Successful v2 carts do not hit this error (E5, E26). The share of failing requests matching the v1 cart share (~22%, E6) is a correlation that has not been confirmed per request, and the log count is much lower than the error rate, so the exact failing population is not fully established.
  - judge: The report identifies the same originating change and mechanism as the key: checkout-svc v2.14.0 changed formatPrice to read cart.currency.code, which is undefined on legacy v1 carts and throws a TypeError. That matches the answer key's checkout-svc v2.14.0 / src/pricing/format.ts failure path.
  - claims no tool result supports: ['"orders-db latency change"', '"No infra events, payments-svc 5xx (flat at 0.1%), or orders-db latency change coincide with the onset"']
- **inc-003/baseline** (correct; lexical correct, alias match 1.0): Feature flag pdp_cache_include_reviews was ramped to 100% at 18:30Z, so PDP cache entries embed the latest 200 reviews. This inflated cached values roughly 9x. Redis reached its 8 GB maxmemory at ~19:05Z, and allkeys-lru eviction collapsed the hit ratio to ~0.40. Cache misses then hit catalog-db's uncached reviews query, which saturated the DB (CPU ~98%) and the catalog-svc connection pool. Gateway upstream timeouts to catalog-svc produced the 504s and the p99 at the 5 s ceiling.
  - judge: The report identifies the same originating change and mechanism as the key: enabling `pdp_cache_include_reviews` inflated PDP cache entries, filled Redis, caused mass eviction, collapsed cache hit ratio, and overloaded catalog-db until catalog-svc timed out behind the gateway.
  - claims no tool result supports: ['"Cached product:pdp entries grew from ~4.1 KB to ~38 KB"', '"gateway itself is healthy (CPU 26-31%, falling)"']
- **inc-003/multi_agent** (correct; lexical correct, alias match 1.0): The pdp_cache_include_reviews flag change at 18:30Z (E51) made catalog-svc cache product:pdp entries with the latest 200 reviews embedded, raising cached PDP values from about 4.1 KB to about 38 KB (E11, E42). Redis hit maxmemory at 19:05Z, causing evictions and a hit-ratio collapse to about 0.40 (E9, E23, E10). The resulting cache-miss load, with reviews fetched from catalog-db, saturated catalog-db CPU and exhausted the catalog-svc DB pool (E12, E13, E18, E30). catalog-svc p99 then reached 5000 ms, and the gateway timed out upstream calls and returned 504s (E4, E2, E29). The flag cause rests on change-record and timing evidence and has not been confirmed from a flag-state source.
  - judge: It identifies the same originating change and mechanism as the key: enabling pdp_cache_include_reviews enlarged PDP cache entries, pushed Redis to maxmemory, caused evictions/hit-ratio collapse, and overloaded catalog-db so catalog-svc timed out behind the gateway.
  - claims no tool result supports: ['"cached PDP values from about 4.1 KB to about 38 KB"', '"Gateway CPU stayed at 26-32% throughout"']
- **inc-004/baseline** (correct; lexical correct, alias match 1.0): Config change PR #874 (identity-team, 11:40Z) lowered auth-svc AUTH_DB_POOL_MAX from 50 to 5 instead of setting minimum idle. The HikariCP pool (max=5) starved under normal login load, which produced queueing and 30s connection timeouts on POST /login.
  - judge: The report identifies the same originating change and mechanism as the key: PR #874 at 11:40Z changed auth-svc's AUTH_DB_POOL_MAX from 50 to 5, causing Hikari pool starvation and login timeouts.
  - claims no tool result supports: ['"the auth-svc restart log is at 11:41:05Z"']
- **inc-004/multi_agent** (correct; lexical correct, alias match 1.0): PR #874 'Tune DB pool' (identity-team, config change at 2026-10-06T11:40:00Z) set AUTH_DB_POOL_MAX from 50 to 5 in auth-svc. With the cap at 5, auth-svc could run only 5 concurrent DB connections while login demand was about 220 rps. Requests queued for a connection (pending threads rose to about 140-151), waited up to the 30 s pool timeout, and the gateway then logged upstream auth-svc timeouts. This is internal pool exhaustion, not a network or users-db fault.
  - judge: The report identifies the same originating change and mechanism as the key: PR #874 changed auth-svc's AUTH_DB_POOL_MAX from 50 to 5, causing Hikari pool exhaustion and login timeouts.
  - claims no tool result supports: ['"edge and Cloudflare telemetry was never queried"']
- **inc-005/baseline** (correct; lexical correct, alias match 1.0): Degraded EBS volume vol-0ab3f9c1 (eu-west-1c) attached to node ip-10-0-3-17. Disk I/O await rose from about 2ms to about 190ms, causing slow local RocksDB stock reads on inventory-svc pods xk2, p9t and m4c. Those reads exceed checkout-svc's 3000ms reserveInventory timeout, so checkout returns AVAILABILITY_UNCONFIRMED to shoppers. Affected traffic is roughly one-sixth of inventory calls (3 of 18 pods), which is why failures are intermittent and why the global dashboards look normal.
  - judge: The report identifies the same primary cause as the key: degraded EBS volume vol-0ab3f9c1 on node ip-10-0-3-17, producing high disk latency that made inventory-svc RocksDB reads slow enough to time out checkout requests.
  - claims no tool result supports: ['"the average latency is dominated by the 15 healthy pods"', '"the node\'s pods still pass readiness"']
- **inc-005/multi_agent** (review; lexical correct, alias match 0.667): Probable mechanism at medium confidence: RocksDB reads in inventory-svc pods on node ip-10-0-3-17 (confirmed for pod inventory-svc-5d8-m4c) became very slow from about 10:20Z, with rocksdb.get taking about 3.8s and node p99 about 4.6s. Those reads exceeded checkout-svc's 3000ms timeout on reserveInventory, so checkout returned AVAILABILITY_UNCONFIRMED. Only the requests routed to the affected pods fail, which is why the aggregate error rate is about 1% and average latency stays flat. The underlying trigger of the RocksDB slowness on that node is not established; no node resource metrics or infra events confirm a node-level fault.
  - judge: The report correctly identifies node ip-10-0-3-17 and slow RocksDB reads as the immediate mechanism behind the checkout timeouts, but it does not name the key originating change: a degraded EBS volume behind that node. It treats the node fault as unproven rather than the root cause.
  - claims no tool result supports: ['"readiness stayed at 100%"', '"Only the requests routed to the affected pods fail"']
- **inc-007/baseline** (correct; lexical correct, alias match 1.0): Automated PSP key rotation (vault rotate-psp-keys, configured 2026-05-25 with revoke_previous_immediately=true) revoked the previous live key pk_live_..A7 at the PSP at 07:00:30Z while payments-svc pods were still running with it. Pods did not reload the new key pk_live_..C2 from vault, so their PSP authorize calls were rejected with 401 invalid_api_key.
  - judge: The report identifies the same originating change and mechanism as the key: the PSP key rotation revoked the old live key immediately, and payments-svc kept using that revoked key because pods did not reload credentials, so card authorizations returned 401 invalid_api_key and were surfaced as PAYMENT_METHOD_UNVERIFIED.
  - claims no tool result supports: ['"Pods did not reload the new key pk_live_..C2 from vault"', '"The one pod that loaded C2 (created by an HPA scale-up at 09:08Z) authorizes successfully"', '"the failure is not visible in 5xx or healthz metrics"', '"/healthz" stayed at 100% during the outage']
- **inc-007/multi_agent** (correct; lexical correct, alias match 1.0): The vault rotation job revoked the previous PSP key pk_live_..A7 immediately at 07:00:30Z (under the 2026-05-25 change setting revoke_previous_immediately=true). payments-svc kept authorizing card payments with the cached A7, so the PSP returned 401 invalid_api_key, which payments-svc mapped to PAYMENT_METHOD_UNVERIFIED. It did not pick up the newly created C2 until one pod loaded it at 09:08, and card success recovered only partially after that. UPI and wallets use other credentials and were unaffected. The per-pod A7-to-C2 switch and the caching behaviour are not directly confirmed by logs.
  - judge: The report identifies the same originating change and mechanism as the key: immediate PSP key revocation during rotation, with payments-svc continuing to use the revoked key and getting 401 invalid_api_key responses that were mapped to a user-facing payment verification failure. That matches the key’s cause and impact.
  - claims no tool result supports: ['"cached A7" / "payments-svc kept authorizing card payments with the cached A7"']
- **inc-008/baseline** (correct; lexical correct, alias match 1.0): (no root cause)
  - judge: The report’s rank-1 hypothesis is a poison message tied to the checkout-svc v4.1.0 gift-wrap change, which matches the key’s originating checkout-svc schema change and decode-failure mechanism. It remains inconclusive rather than claiming a different primary cause.
  - claims no tool result supports: ['"Orders on that partition never reach orders-db or notification-svc"']
- **inc-009/baseline** (correct; lexical correct, alias match 1.0): Single data-node hardware failure (es-data-3, 15:17Z) on an Elasticsearch index created with zero replicas (change 2026-09-16 "Reindex products into new cluster with number_of_replicas=0", follow-up SRCH-311 never completed). Two of the 10 primary shards of 'products' became unassigned, so search-svc's queries against them failed or returned partial results.
  - judge: The report identifies the same primary cause as the key: es-data-3 failed, and because the products index had been left at number_of_replicas=0, two primary shards became unassigned and searches returned partial or failed results. It matches both the originating change and the mechanism.
  - claims no tool result supports: ['"footer-only change" for web-frontend v12.3.0', '"detected by users first"']
- **inc-010/baseline** (correct; lexical correct, alias match 1.0): The api HPA maxReplicas was reduced from 40 to 12 on 2026-09-29 (FinOps cost review, call-0007). When the 2026-10-02 20:00Z 'Diwali flash sale' push produced ~6x traffic, the HPA requested 34 replicas but was capped at 12 (ScalingLimited, call-0004). The 12 pods saturated and the request queue (max 200) rejected traffic with 503s (call-0017). The campaign checklist has no pre-scaling step (call-0015), and the Sept 5 campaign had already needed 31 replicas (call-0014, call-0015), so the 12-replica cap was unsafe for known campaign load.
  - judge: The report identifies the same combined cause as the key: a campaign-driven traffic surge from the Diwali flash sale, together with the 09-29 FinOps HPA cap reduction to 12 replicas that prevented the API from scaling to the needed level.
  - claims no tool result supports: ['"the 12 pods saturated"']
- **inc-010/multi_agent** (correct; lexical correct, alias match 1.0): The api HorizontalPodAutoscaler had its maxReplicas lowered from 40 to 12 by the 2026-09-29 FinOps cost-review change (E10). When the Diwali flash sale push went out at 20:00Z and api traffic rose about 6x (E1, E20), the HPA computed a desired count of 34 but could not scale past the cap of 12 (E11, E30). The 12 replicas saturated at about 96% CPU (E5, E21, E31). Their request queues filled (max 200), and they rejected requests with 503s (E3, E4, E22), which the ALB passed through as 503s (E4). The traffic surge itself is best explained as legitimate campaign demand (E2, E6, E12, E16, E28), not an external DDoS, though source-level confirmation is still missing.
  - judge: The report identifies the same combined cause as the key: a Diwali campaign traffic surge and the FinOps change that capped api HPA maxReplicas at 12, preventing the service from scaling to the 34 replicas it wanted. It also correctly treats the DDoS and orders-db ideas as non-causes.
  - claims no tool result supports: ['"no related errors or alerts" for orders-db']
- **inc-011/multi_agent** (correct; lexical correct, alias match 1.0): Migration 0047_users_bio_to_text ran ALTER COLUMN TYPE on users.bio at 16:00Z, which rewrote the users table in users-db. The rewrite held a lock that blocked reads and writes on users (lock_wait_sessions rose to about 180-251 from 16:00Z to 16:20Z). user-svc queries on GET and PUT /users/{id}/profile then waited past the lock timeout (about 5 seconds), and user-svc returned 5xx. Load fell back to baseline by about 16:25Z, when lock waits cleared. The lock mode and exact lock duration are not confirmed by change or infra data.
  - judge: The report pins the outage on migration 0047_users_bio_to_text rewriting the users table during ALTER COLUMN TYPE, which blocked reads and writes until lock timeouts cleared. That matches the key’s originating change and mechanism.
  - claims no tool result supports: ['"user-svc logs appear sampled"']
- **inc-012/multi_agent** (review; lexical correct, alias match 0.8): Mechanism supported by the evidence: AI and web crawlers, which self-identify as crawlers and come from cloud IP ranges, made a large increase in requests to nginx-edge starting 2026-04-29 (E1, E9, E25, E27). Because nginx-edge has no CDN in front and no rate limiting (E15, E28), every request reached the EC2 host and was served as full responses. Unknown paths return 200 index.html through the try_files fallback, and static JS and blog HTML were fetched repeatedly, so egress scaled with crawler request volume at about 0.8 MB per request (E18, E26, E8). The EC2 DataTransfer-Out cost followed that volume (E23). The initial trigger of the crawler increase was not identified, and the day-by-day match is only confirmed for 05-03.
  - judge: The report correctly centers AI/web crawler traffic hitting nginx-edge and the unbounded crawl/fallback/no-CDN egress mechanism. However, it does not identify the specific originating change the key names — permissive robots.txt / crawl authorization starting the crawl explosion — instead leaving the trigger unidentified.
- **inc-013/baseline** (correct; lexical correct, alias match 1.0): Our outbound dispatch (campaign-scheduler new attempts at 12/sec plus retries, with no global cap against the carrier's 15 attempts/sec contract) exceeded the SIP trunk's call-attempt rate. The carrier answered the excess INVITEs with 429. The dialer counted each 429 as a failed attempt, and after 4 attempts the scheduler marked the contacts UNREACHABLE without ever placing a call.
  - judge: The report identifies the same component and mechanism as the key: the campaign-scheduler/dialer retry behavior causes carrier call-attempt-rate overruns, 429 rejections, and UNREACHABLE marks. It does not hinge on the channel limit or a pure carrier outage.
  - claims no tool result supports: ['"when the 1-minute average was ~11.5/sec"']
- **inc-013/multi_agent** (wrong; lexical correct, alias match 0.8): The K-2214 campaign start at 2026-03-10T04:00Z raised INVITE attempt volume on the carrier trunk above the carrier's enforced limit of 15 call attempts/second (all INVITEs counted, including retries). The carrier rejected the excess INVITEs with SIP 429 from about 05:00Z (E7, E8), and sip-gateway passed those errors back to the dialer (E12, E13). The step that marked contacts UNREACHABLE without a call reaching the phone is not yet confirmed (E3, H3 inconclusive), and the onset date should be re-checked against the 2026-03-11 report date (E9).
  - judge: The report’s primary cause is a campaign-start-driven overload of the carrier trunk above 15 attempts/sec, whereas the key root cause is a self-sustaining 429 retry storm caused by fixed-interval retries that synchronize into waves. It mentions retries and backoff as follow-up fixes, but not as the originating mechanism.
  - claims no tool result supports: ['"The K-2214 campaign start at 2026-03-10T04:00Z raised INVITE attempt volume on the carrier trunk above the carrier\'s enforced limit of 15 call attempts/second"', '"The step that marked contacts UNREACHABLE without a call reaching the phone is not yet confirmed"', '"The onset date should be re-checked against the 2026-03-11 report date"']
- **inc-014/baseline** (review; lexical correct, alias match 1.0): pricing-cache-001 (primary) failed on host hardware fault at 13:17:42Z. ElastiCache promoted pricing-cache-002, which had only 0.4M keys versus 9.8M on the old primary (its replica had gone into a full resync at 13:02 after its replication backlog was exceeded). The resulting cold cache (hit ratio ~0.15) pushed most quote lookups to rates-db, which saturated. Pricing quote timeouts then exceeded search-api's 2500ms deadline, producing high p99 latency, errors and partial ranking. The warmer step that the runbook prescribes after a failover was not run.
  - judge: The report identifies the right component and mechanism: pricing-cache failover to a nearly empty replica caused the cold-cache miss storm. But it frames the primary root cause as a host hardware fault on pricing-cache-001, whereas the key’s root cause is the failover to the replica that was mid full-resync.
  - claims no tool result supports: ['The warmer step that the runbook prescribes after a failover was not run.']
- **inc-016/baseline** (correct; lexical correct, alias match 1.0): Backfill job backfill-carrier-events (data-jobs, started 08:49:50 UTC with parallelism 16, against runbook limits) saturated tracking-db writes and locks. Tracking-api DB queries slowed, connections were held longer, and the pool exhausted. The reduction of the pool to 40 at 08:55 narrowed the ceiling and turned the slowdown into 5xx errors (pool acquire timeouts).
  - judge: The report identifies the same originating component and mechanism as the key: the backfill-carrier-events job overloaded tracking-db with heavy writes/locks, which drove query latency up and exhausted tracking-api’s connection pool. The pool reduction is treated as a contributing factor, not the primary cause.
  - claims no tool result supports: ['“pool usage had already risen to 31”']
- **inc-016/multi_agent** (correct; lexical correct, alias match 1.0): The backfill-carrier-events Kubernetes job (data-jobs, created 08:49:50Z, parallelism 16) ran INSERT ... ON CONFLICT DO UPDATE batches of 5,000 rows against shipment_events on tracking-db. This drove write IOPS to about 98% of the provisioned 10,000 and produced row-lock waits (Lock:tuple) on shipment_events. Tracking-api SELECTs on shipment_events were blocked on those locks while holding pooled connections, so connection hold times grew and pool in-use rose from 18 to 31 by 08:50. When db.pool.max_size was cut from 50 to 40 at 08:55:20Z, the pool was exhausted (in-use pinned at 40, wait p99 about 1,950 ms). Requests then timed out acquiring a connection after 2000 ms and returned 5xx. The blocking session is inferred from timing and the trace evidence, not confirmed by lock-holder data.
  - judge: The report identifies the backfill-carrier-events job on tracking-db as the origin and explains the same mechanism as the key: heavy write load and lock contention drove tracking-db latency up, held tracking-api connections longer, and exhausted the pool. It also treats the later pool-size cut as an amplifier, not the primary cause.
  - claims no tool result supports: ['"pool wait p99 rose to about 1,950 ms"']

## Metric definitions

- **root_cause_accuracy**: share of runs whose root cause the judge grades as the answer key's root cause (not merely mentioned as an amplifier or trigger); for expected-inconclusive scenarios, share that reported inconclusive. `review` = judge "partial" (or, without a judge, a partial alias match), decided manually in `reviews.yaml`. **root_cause_accuracy_lexical**: the same by alias word overlap only.
- **fell_for_red_herring / must_not_violations**: runs that adopt a red herring as cause or primary fix / do something the key forbids (judge).
- **runs_with_unsupported_claims**: runs whose report states as observed a specific fact (value, setting, event) that no tool result in the run shows (judge, checked against the saved tool data).
- **claims_grounded**: share of supported/refuted hypotheses and corrective actions citing at least one evidence id that traces to an executed tool call. **citations_valid**: share of all cited ids that do.
- **key_evidence_retrieved / cited**: share of the answer key's required evidence the system fetched (a successful call whose returned data contains the evidence) / cited in its report.
- **factors_mentioned**: share of must-mention contributing factors the report states (judge; lexical word overlap without a judge).
- Tokens include cache reads/writes; cost uses `investigator/llm/pricing.py`. Latency is wall clock per run.
