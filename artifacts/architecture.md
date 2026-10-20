# Architecture (Mermaid)

Based on `planning.md` and `initial_architecture.png`, with the agreed review decisions applied: orchestrator as the single writer, three parallel gatherers, and critic loops routed through the orchestrator. The single-agent baseline (section 4) is the comparison system. The evaluator runs offline in `evals/`, outside the graph.

Mapping to the brief's roles: Orchestrator = `triage` (LLM planner) + `orchestrator` (deterministic manager); Observability Agent = `observability`; Dependency Agent = `dependency` (service dependencies) together with `change_history` (deployments and configuration changes), split so each keeps a short, focused toolset; Reasoning, Critic and Reporting Agents = `reasoning`, `critic`, `report`.

## 1. Agent graph

```mermaid
flowchart TD
    IN(["Incident report"]) --> PLAN

    PLAN["ORCHESTRATOR: LLM planner<br/>sees topology, recent changes in all services,<br/>each specialist's toolset<br/>-> hypotheses + tasks with dependencies"]
    PLAN --> ORCH

    ORCH{{"ORCHESTRATOR: deterministic manager<br/>single writer of shared state<br/>applies outcomes, task dependencies,<br/>budgets, evidence gate, routing"}}

    ORCH -- "Send(task), in parallel" --> OBS["OBSERVABILITY<br/>metrics, logs, traces,<br/>alerts, infra events"]
    ORCH -- "Send(task)" --> DEP["DEPENDENCY<br/>topology, blast radius,<br/>dependency health"]
    ORCH -- "Send(task)" --> CHG["CHANGE HISTORY<br/>always starts with an unfiltered change sweep;<br/>deploys, config, flags, past incidents, runbooks"]

    OBS & DEP & CHG -- "tool calls" --> TH[["Tool harness<br/>validation, timeout, retries + backoff,<br/>simulated faults, data-quality flags"]]
    OBS & DEP & CHG -- "Outcome: evidence citing own call_ids" --> MB[("Outcomes mailbox<br/>append-only")]
    MB -- "applied in task order" --> ORCH

    ORCH -- "evidence snapshot" --> REA["REASONING<br/>rank competing hypotheses,<br/>supporting / contradicting evidence"]
    REA -- "Outcome: ranking" --> MB
    ORCH -- "ranking + budget left" --> CRI["CRITIC<br/>accept / need more evidence /<br/>inconclusive (must name the blocking gap)"]
    CRI -- "Outcome: verdict" --> MB
    CRI -. "need more evidence:<br/>targeted tasks, max 3 rounds" .-> ORCH

    ORCH -- "limit hit (tool calls, tokens, time)" --> FIN["Closing pass<br/>rank what was gathered, critic decides;<br/>no more gathering"]
    FIN --> GATE
    ORCH -- "critic accepts / inconclusive" --> GATE{"Evidence gate<br/>top hypotheses have evidence,<br/>every specialist ran"}
    GATE -- "missing evidence" --> ORCH
    GATE -- "pass" --> REP["REPORTING<br/>ranked hypotheses, facts vs assumptions,<br/>confidence, corrective actions"]
    REP --> OUT(["Final report"])

    ORCH -. "needs human (interactive)" .-> HUM[/"Human review<br/>interrupt + checkpoint resume"/]
    HUM -. "answer" .-> ORCH

    OUT -.-> EVAL["EVALUATOR (offline, evals/)<br/>answer keys + LLM judge"]
    classDef offline stroke-dasharray: 5 5,color:#888
    class EVAL offline
```

**Critic verdicts**, all applied by the orchestrator:
- `need_more_evidence(requests)`: new targeted tasks for the gatherers, at most 3 loops. The critic sees which
  specialist holds which tools and the remaining budget, and only asks questions a tool can answer.
- `needs_human`: interactive mode interrupts. In eval mode the critic is told no human is available.
- `accept` or `inconclusive`: go to the evidence gate. `inconclusive` must name a blocking gap or an equally
  supported competing hypothesis; otherwise the orchestrator treats it as accept (and logs that).

**How the agents collaborate**:
- The planner sees the topology, an unfiltered list of recent changes in all services and each specialist's
  toolset, and turns the incident into hypotheses and tasks with dependencies.
- Specialists work in parallel on their own short contexts. Each brief carries the hypotheses under test, the
  team's evidence so far and every query already run, so re-dispatched work builds on earlier results.
- Every result is a typed `Outcome`. The orchestrator turns outcomes into shared state; evidence that cites a
  tool call the worker did not make is dropped.
- The reasoner ranks hypotheses on the shared evidence; the critic challenges the ranking and sends targeted
  requests back through the orchestrator, which becomes the next round of tasks.

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

The comparison system: one tool-calling agent (`investigator/baseline.py`) with the same `ToolHarness`, tool rendering, budgets, metering and report schema, so the only difference is the orchestration (no planner, specialists, evidence gate or critic). It's a plain ReAct loop (`start -> agent <-> tools -> finalize`, plus `nudge`/`budget` steps that ask it to submit). It's built on our own loop rather than `create_agent` so tool handling and metering are identical to the multi-agent system. Facts in its report must cite tool `call_id`s and are converted into the same evidence index; fabricated call ids are dropped.

```mermaid
flowchart LR
    IN(["Incident report"]) --> SA["SINGLE AGENT<br/>ReAct loop: agent, tools, finalize<br/>all 10 tools + submit_report"]
    SA -- "tool calls" --> H[["ToolHarness<br/>same tools, faults, retries"]]
    H -- "ToolResult" --> SA
    SA --> OUT(["Final report<br/>(same schema)"])

    MA["Multi-agent graph<br/>(section 1)"] -- "tool calls" --> H
```

Both systems record tool calls, tokens and latency per run. That's what the comparison uses: root-cause accuracy, evidence grounding, tool-call count, latency and token consumption.

## 5. Compiled LangGraph (generated)

Generated from the built graph with `build_graph(...).get_graph().draw_mermaid()`. Dashed edges are dynamic `Command`/`Send` routes chosen by the orchestrator. `triage` is the orchestrator's LLM-planner step, kept as its own node so a planner failure can fall back to default coverage.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	triage(triage)
	orchestrator(orchestrator)
	observability(observability)
	dependency(dependency)
	change_history(change_history)
	reasoning(reasoning)
	critic(critic)
	human(human)
	report(report)
	__end__([<p>__end__</p>]):::last
	__start__ --> triage;
	change_history --> orchestrator;
	critic --> orchestrator;
	dependency --> orchestrator;
	human --> orchestrator;
	observability --> orchestrator;
	orchestrator -.-> change_history;
	orchestrator -.-> critic;
	orchestrator -.-> dependency;
	orchestrator -.-> human;
	orchestrator -.-> observability;
	orchestrator -.-> reasoning;
	orchestrator -.-> report;
	reasoning --> orchestrator;
	triage --> orchestrator;
	report --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Termination and limits
| Limit | Default | When hit |
|---|---|---|
| Critic loops | 3 | go to evidence gate, then report with gaps listed |
| Any run limit (tool calls, tokens, wall clock) | see below | one closing pass: rank what was gathered, the critic decides accept or inconclusive; no more gathering. Accepting still needs the evidence gate |
| Tool retries per call | 2 (`TOOL_MAX_RETRIES`) | tool result becomes an evidence gap |
| Tool timeout | 10s (`TOOL_TIMEOUT_S`) | retried, then a gap |
| Tool calls per run | 60 (`BudgetLimits.max_tool_calls`) | stop gathering, closing pass, report |
| Wall clock / tokens | 600s / 400k tokens | stop gathering, closing pass, report |
| Tasks per round | 6 | extra planner/critic requests dropped (logged) |
| Tool calls per task | 8 | further calls return `budget_exhausted` |
| Worker attempts | 2 | then `Outcome(failed)`; investigation continues |
| Evidence gate | top 3 hypotheses need evidence + all 3 gatherers ran | targeted gap tasks while loops remain |
| LangGraph `recursion_limit` | backstop | hard stop |
