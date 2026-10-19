# Architecture (Mermaid)

Based on `planning.md` and `initial_architecture.png`, with the agreed review decisions applied: orchestrator as the single writer, three parallel gatherers, and critic loops routed through the orchestrator. The **single-agent baseline is required**. Only the **evaluator** is optional/deferred.

## 1. Agent graph

```mermaid
flowchart TD
    IN(["Incident report"]) --> ORCH

    ORCH{{"ORCHESTRATOR<br/>LLM planner: triage, initial hypotheses, tasks<br/>Deterministic manager: apply outcomes, budgets, routing<br/>(only writer of shared state)"}}

    ORCH -- "Send(task), parallel<br/>initial + targeted re-dispatch" --> OBS["OBSERVABILITY<br/>metrics, logs, traces,<br/>alerts, infra events"]
    ORCH -- "Send(task)" --> DEP["DEPENDENCY<br/>topology, blast radius"]
    ORCH -- "Send(task)" --> CHG["CHANGE & HISTORY<br/>deploys, config, flags, migrations,<br/>past incidents, runbooks"]

    OBS -- "Outcome" --> MB[("Outcomes mailbox<br/>append-only")]
    DEP -- "Outcome" --> MB
    CHG -- "Outcome" --> MB
    MB --> ORCH

    ORCH -- "evidence snapshot" --> REA["REASONING<br/>reasoning LLM: rank hypotheses,<br/>supporting / contradicting evidence"]
    REA -- "Outcome: ranked hypotheses" --> MB

    ORCH -- "hypotheses + evidence" --> CRI["CRITIC<br/>challenge conclusions against evidence"]
    CRI -- "Outcome: verdict" --> MB

    ORCH -- "done or limit" --> GATE{"Evidence gate<br/>each top hypothesis has<br/>support or contradiction?"}
    GATE -- "pass or limit reached" --> REP["REPORTING<br/>ranked hypotheses, evidence ids, confidence,<br/>facts vs assumptions, actions"]
    GATE -- "missing evidence" --> ORCH
    REP --> OUT(["Final report"])

    ORCH -. "needs_human" .-> HUM[/"Human review<br/>interrupt + checkpoint"/]
    HUM -. "resume with input" .-> ORCH

    OUT -.-> EVAL["EVALUATOR (optional, offline)"]

    classDef optional stroke-dasharray: 5 5,color:#888
    class EVAL optional
```

**Critic verdicts**, all applied by the orchestrator:
- `need_more_evidence(requests)`: new targeted tasks for the gatherers, at most 3 loops.
- `needs_human`: interactive mode interrupts; eval mode produces an inconclusive report with the open questions.
- `accept` or `inconclusive`: go to the evidence gate.

## 2. Shared state, single writer

```mermaid
flowchart LR
    subgraph Workers["Workers (read-only task payload)"]
        W1["Observability"]
        W2["Dependency"]
        W3["Change & History"]
        W4["Reasoning"]
        W5["Critic"]
    end

    Workers -- "append Outcome<br/>(list reducer, safe in parallel)" --> MB[("outcomes")]

    MB --> ORCH{{"Orchestrator<br/>apply outcomes in order"}}

    ORCH --> S1["tasks + depends_on"]
    ORCH --> S2["evidence (ids, source call, quality flags)"]
    ORCH --> S3["hypotheses (ranked)"]
    ORCH --> S4["critic verdicts + loop count"]
    ORCH --> S5["budget: tool calls, tokens, wall clock"]
    ORCH --> S6["gaps + errors"]
    ORCH --> S7["phase"]

    subgraph Shared["Shared investigation state (orchestrator-owned)"]
        S1
        S2
        S3
        S4
        S5
        S6
        S7
    end
```

## 3. Tool call path, reliability

```mermaid
sequenceDiagram
    participant A as Gatherer agent
    participant H as ToolHarness
    participant F as Fault injector
    participant B as Simulated backend<br/>(Sentry-style logs, metrics,<br/>GitHub-style deploys, ...)

    A->>H: call(tool, args)
    H->>H: validate input (Pydantic)
    alt invalid args / unknown service
        H-->>A: ToolResult(ok=false, invalid_request / not_found, no retry)
    else valid
        loop attempt 1..(1 + TOOL_MAX_RETRIES)
            H->>F: active fault for this call?
            H->>B: query (per-call timeout)
            alt timeout / unavailable (retryable)
                H->>H: sleep(exp backoff + jitter)
            else success
                B-->>H: typed output
            end
        end
        H->>H: data-quality checks<br/>(range vs requested, empty, gaps, at limit)
        H-->>A: ToolResult(ok, data, quality, attempts, latency)
    end
```

## 4. Single-agent baseline (required)

The comparison system: one tool-calling agent with the same `ToolHarness`, tools, budgets and report schema, so the only difference is the orchestration.

```mermaid
flowchart LR
    IN(["Incident report"]) --> SA["SINGLE AGENT<br/>one tool-calling LLM loop<br/>(create_agent)"]
    SA -- "tool calls" --> H[["ToolHarness<br/>same tools, faults, retries"]]
    H -- "ToolResult" --> SA
    SA --> OUT(["Final report<br/>(same schema)"])

    MA["Multi-agent graph<br/>(section 1)"] -- "tool calls" --> H
```

Both systems record tool calls, tokens and latency per run. That's what the comparison uses: root-cause accuracy, evidence grounding, tool-call count, latency and token consumption.

## Termination and limits
| Limit | Default | When hit |
|---|---|---|
| Critic loops | 3 | go to evidence gate, then report with gaps listed |
| Tool retries per call | 2 (`TOOL_MAX_RETRIES`) | tool result becomes an evidence gap |
| Tool timeout | 10s (`TOOL_TIMEOUT_S`) | retried, then a gap |
| Tool calls per run | 60 (`BudgetLimits.max_tool_calls`) | stop gathering and report |
| Wall clock / tokens | 600s / 400k tokens | stop gathering and report |
| Tasks per round | 6 | extra planner/critic requests dropped (logged) |
| Evidence gate | top 3 hypotheses need evidence + all 3 gatherers ran | targeted gap tasks while loops remain |
| LangGraph `recursion_limit` | backstop | hard stop |
