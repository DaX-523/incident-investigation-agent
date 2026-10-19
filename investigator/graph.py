"""LangGraph wiring: orchestrator hub, parallel gatherers via Send, reasoning, critic, human review, report.

Worker implementations are injected (`Workers`), so the same graph runs with LLM
agents or with deterministic stubs (tests, offline dry runs).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send, interrupt

from investigator.orchestrator.context import WorkerContext, WorkerInput, build_context
from investigator.orchestrator.manager import apply_triage, step
from investigator.orchestrator.report_builder import build_report
from investigator.report import ReportNarrative
from investigator.schemas import (
    GATHERERS,
    AgentName,
    HypothesisSpec,
    InvestigationState,
    Outcome,
    Task,
    TriagePlan,
    Usage,
    failed_outcome,
)
from investigator.llm.metering import UsageLedger
from investigator.tools.harness import ToolHarness
from investigator.tools.toolbox import ToolBox

logger = logging.getLogger(__name__)


class Planner(Protocol):
    def __call__(self, incident: dict, toolbox: ToolBox) -> TriagePlan: ...


class Gatherer(Protocol):
    def __call__(self, task: Task, context: WorkerContext, toolbox: ToolBox) -> Outcome: ...


class Analyst(Protocol):
    def __call__(self, task: Task, context: WorkerContext) -> Outcome: ...


class Reporter(Protocol):
    def __call__(self, state: InvestigationState) -> ReportNarrative: ...


@dataclass
class Workers:
    planner: Planner
    gatherers: dict[str, Gatherer]
    reasoner: Analyst
    critic: Analyst
    reporter: Reporter | None = None


@dataclass
class RunContext:
    harness: ToolHarness
    clock: Callable[[], float] = field(default=time.time)
    # LLM token/cost ledger shared with the workers (None for LLM-free stubs).
    ledger: UsageLedger | None = None
    # Workers resolved per run (used when build_graph() was given none, e.g. LangGraph Studio).
    workers: Workers | None = None


def _charge(budget, agent: str, usage: Usage):
    by_agent = dict(budget.by_agent)
    by_agent[agent] = by_agent.get(agent, Usage()) + usage
    return budget.model_copy(update={"usage": budget.usage + usage, "by_agent": by_agent})


# Tool calls the planner may make (e.g. get_topology) before any task exists.
PLANNER_TOOL_ALLOWANCE = 3


def fallback_plan(incident: dict) -> TriagePlan:
    """Used when the planner fails: the reporter's claim as the only hypothesis; coverage sweeps fill in tasks."""
    claim = f"{incident.get('title', '')}: {incident.get('description', '')}".strip(": ")
    return TriagePlan(
        summary="Planner unavailable; investigating with default coverage.",
        hypotheses=[HypothesisSpec(key="reporter", statement=f"Reporter's account: {claim[:300]}", origin="reporter")],
        tasks=[],
    )


def build_graph(
    workers: Workers | None = None,
    checkpointer=None,
    *,
    context_schema: type = RunContext,
    bootstrap: Callable[[Any], dict] | None = None,
):
    """Compile the investigation graph.

    `workers=None` resolves workers from the run context per run. `bootstrap(context)`, if given,
    adds an entry node that builds the initial state when the input has no incident (Studio).
    """

    def _workers(runtime: Runtime[RunContext]) -> Workers:
        resolved = workers or runtime.context.workers
        if resolved is None:
            raise RuntimeError("no workers: pass them to build_graph() or provide them in the run context")
        return resolved

    def load_incident(state: InvestigationState, runtime: Runtime[RunContext]) -> dict:
        return {} if state.get("incident") else bootstrap(runtime.context)

    def triage(state: InvestigationState, runtime: Runtime[RunContext]) -> dict:
        toolbox = ToolBox(runtime.context.harness, PLANNER_TOOL_ALLOWANCE)
        try:
            plan = _workers(runtime).planner(state["incident"], toolbox)
            errors: list[str] = []
        except Exception as exc:  # recovery: never fail the run on the planner
            logger.warning("planner failed: %s", exc)
            plan, errors = fallback_plan(state["incident"]), [f"planner failed: {type(exc).__name__}: {exc}"]
        update = apply_triage(state, plan)
        ledger = runtime.context.ledger
        planner_usage = (ledger.total("triage") if ledger else Usage()) + Usage(tool_calls=len(toolbox.records()))
        update["budget"] = _charge(update["budget"], "triage", planner_usage)
        update["errors"] = update["errors"] + errors
        return update

    def orchestrator(state: InvestigationState, runtime: Runtime[RunContext]) -> Command:
        decision = step(state, now=runtime.context.clock())
        if decision.action == "dispatch":
            merged = InvestigationState(**{**state, **decision.update})
            goto = [Send(t.agent, WorkerInput(task=t, context=build_context(merged, t))) for t in decision.tasks]
        else:
            goto = decision.action  # "human" | "report"
        return Command(update=decision.update, goto=goto)

    def make_worker(agent: AgentName):
        def run_worker(inp: WorkerInput, runtime: Runtime[RunContext]) -> dict:
            task, context = inp["task"], inp["context"]
            max_attempts = max(1, context.max_worker_attempts)
            toolbox = ToolBox(runtime.context.harness, context.tool_call_allowance)
            last_error: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    w = _workers(runtime)
                    if agent in GATHERERS:
                        outcome = w.gatherers[agent](task, context, toolbox)
                    elif agent == "reasoning":
                        outcome = w.reasoner(task, context)
                    else:
                        outcome = w.critic(task, context)
                    # Tool calls are recorded by the toolbox, not trusted from the worker.
                    outcome = outcome.model_copy(update={"task_id": task.task_id, "agent": agent,
                                                         "tool_calls": toolbox.records()})
                    return {"outcomes": [outcome]}
                except Exception as exc:  # bounded worker retry, then a failed outcome
                    last_error = exc
                    logger.warning("%s worker failed on %s (attempt %d/%d): %s",
                                   agent, task.task_id, attempt + 1, max_attempts, exc)
            failed = failed_outcome(task, f"{type(last_error).__name__}: {last_error} (after {max_attempts} attempts)")
            return {"outcomes": [failed.model_copy(update={"tool_calls": toolbox.records()})]}

        return run_worker

    def human_review(state: InvestigationState) -> dict:
        verdict = state["verdicts"][-1] if state.get("verdicts") else None
        answer = interrupt({
            "questions": verdict.questions_for_human if verdict else [],
            "issues": verdict.issues if verdict else [],
        })
        return {"human_inputs": [str(answer)]}

    def report(state: InvestigationState, runtime: Runtime[RunContext]) -> dict:
        narrative = None
        errors = list(state["errors"])
        ledger = runtime.context.ledger
        reporter = _workers(runtime).reporter
        if reporter is not None:
            try:
                narrative = reporter(state)
            except Exception as exc:
                logger.warning("reporter failed: %s", exc)
                errors.append(f"reporter failed: {type(exc).__name__}: {exc}; deterministic report used")
        budget = _charge(state["budget"], "reporter", ledger.total("reporter")) if ledger else state["budget"]
        return {"report": build_report(state, narrative), "phase": "done", "errors": errors, "budget": budget}

    graph = StateGraph(InvestigationState, context_schema=context_schema)
    graph.add_node("triage", triage)
    graph.add_node("orchestrator", orchestrator,
                   destinations=(*GATHERERS, "reasoning", "critic", "human", "report"))
    for agent in (*GATHERERS, "reasoning", "critic"):
        graph.add_node(agent, make_worker(agent), input_schema=WorkerInput)
        graph.add_edge(agent, "orchestrator")
    graph.add_node("human", human_review)
    graph.add_node("report", report)

    if bootstrap is not None:
        graph.add_node("load_incident", load_incident)
        graph.add_edge(START, "load_incident")
        graph.add_edge("load_incident", "triage")
    else:
        graph.add_edge(START, "triage")
    graph.add_edge("triage", "orchestrator")
    graph.add_edge("human", "orchestrator")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)

