#hand written append only log doc for initial planning, key decisions and artifacts

INCIDENT INVESTIGATOR AGENT

- PROVISIONS(sentry & github) -> access to simulated logs, performance metrics, deployment history, service dependencies, historical incidents reports

- CONSTRAINT -> data may be incomplete and misleading and contradicting, no hardcoded test cases, frontend optional/minimal

- EVALUATION -> At least 10 test incidents on misleading  evidence/conflictingf observability/tool failure

some key decisions after initial stage
- the orchestrator looks helpless, so now i think rather giving CRUD ops access to each nodes, i should give authority to orchestrator node. Orchestrator will gain knowledge from node and then takes action
in this was we can avoid race condition when parallel execution is done (layer 1 specially)

This will make the graph look changed a bit

- evaluator agent is placed in optional for now, i will probably introduce it later, but not focusing on it first

Architecture review decisions 

- PROVISIONS: Sentry/GitHub are SIMULATED (mock tools shaped like them)
- AGENTS: orchestrator + 3 parallel gatherers on 1st layer (observability: metrics/logs/traces/alerts/infra events; dependency: topology/blast radius; change & history: deploys/config/flags/migrations/past incidents/runbooks) + reasoning + critic + reporting.
- STATE: orchestrator is the SINGLE WRITER of shared state. Workers get a read-only task payload (Send) and return a typed Outcome into an append-only mailbox; orchestrator applies outcomes (no races).
- ORCHESTRATOR: LLM planner (triage, initial hypotheses incl. reporter's claim as unverified, critic requests -> tasks) + deterministic state manager/router (apply outcomes, evidence ids, budgets, task dependencies, next phase).
- FEEDBACK LOOPS: critic -> orchestrator -> targeted re-dispatch, max 3 loops.
- RELIABILITY: tool wrapper with timeout + bounded retries (exp backoff + jitter, retryable errors only), typed ToolResult, data-quality flags (stale/empty/truncated); limits on loops, tool calls, wall clock, tokens; failed worker -> Outcome(failed), never crash.
- HUMAN REVIEW: interrupt + checkpoint resume in interactive mode; in eval mode needs_human -> inconclusive report with questions.
- SINGLE AGENT VS MULTI AGENT COMPARISON

- EVALUATOR (optional): offline evals/ harness (LangSmith + agentevals), not a graph node; compared against single-agent baseline with same tools.
