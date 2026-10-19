"""Single-agent baseline: one ReAct loop with every tool, the same harness, budgets, metering and report schema.

Graph: START -> start -> agent <-> tools -> finalize -> END. No orchestrator, evidence gate or
critic - that difference is what the comparison measures.
"""

from __future__ import annotations

import operator
import time
import uuid
from typing import Annotated, Any, Callable, Literal, Protocol

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import TypedDict

from dataset.schema import Scenario
from investigator.agents import prompts
from investigator.agents.llm_agents import _Caller, _tool_def, render_incident, render_tool_result
from investigator.graph import RunContext
from investigator.llm.metering import UsageLedger
from investigator.report import (
    Claim,
    CorrectiveActionOut,
    EvidenceIndexEntry,
    InvestigationReport,
    ReportHypothesis,
)
from investigator.runner import RunResult
from investigator.schemas import Confidence, InvestigationConfig, StopReason, ToolCallRecord, Usage
from investigator.tools.harness import RetrySettings, ToolHarness
from investigator.tools.registry import TOOL_SPECS
from investigator.tools.toolbox import ToolBox

SUBMIT_REPORT = "submit_report"
MAX_REMINDERS = 2


# --- what the agent submits ---------------------------------------------------------


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaselineFact(_M):
    text: str = Field(description="A concrete observation from tool output.")
    call_ids: list[str] = Field(min_length=1, description="call_ids of the tool results this fact comes from.")


class BaselineHypothesis(_M):
    rank: int
    statement: str
    status: Literal["supported", "refuted", "inconclusive", "open"]
    confidence: Confidence
    supporting: list[BaselineFact] = []
    contradicting: list[BaselineFact] = []
    assumptions: list[str] = []


class BaselineAction(_M):
    rank: int
    category: Literal["mitigate", "fix", "prevent", "detect"]
    action: str
    rationale: str
    call_ids: list[str] = []


class BaselineSubmission(_M):
    outcome: Literal["conclusive", "inconclusive"]
    summary: str
    root_cause: str | None = None
    confidence: Confidence
    hypotheses: list[BaselineHypothesis] = Field(min_length=1)
    corrective_actions: list[BaselineAction] = []
    missing_data: list[str] = []
    security_findings: list[str] = []
    questions_for_human: list[str] = []

    @model_validator(mode="after")
    def _conclusive_needs_supported_top(self) -> BaselineSubmission:
        # Same bar the multi-agent evidence gate enforces before its report step.
        if self.outcome == "conclusive":
            top = min(self.hypotheses, key=lambda h: h.rank)
            if top.status != "supported" or not top.supporting:
                raise ValueError("a conclusive report needs its rank-1 hypothesis marked supported, "
                                 "with at least one supporting fact citing call_ids")
        return self


SUBMIT_DEF = {
    "name": SUBMIT_REPORT,
    "description": "Submit your final incident report. Call exactly once, when you are done.",
    "input_schema": BaselineSubmission.model_json_schema(),
}
BASELINE_TOOLS = [_tool_def(name) for name in TOOL_SPECS] + [SUBMIT_DEF]


# --- model adapters ---------------------------------------------------------------------


class BaselineModel(Protocol):
    def __call__(self, messages: list[AnyMessage]) -> tuple[AIMessage, Usage]: ...


class LLMBaselineModel:
    def __init__(self, caller: _Caller):
        self.caller = caller
        self.bound = caller.model.bind_tools(BASELINE_TOOLS)

    def __call__(self, messages):
        return self.caller.invoke(self.bound, messages)


# --- state ---------------------------------------------------------------------------------


class BaselineState(TypedDict, total=False):
    incident: dict[str, Any]
    config: InvestigationConfig
    started_at: float
    messages: Annotated[list[AnyMessage], operator.add]
    tool_records: Annotated[list[ToolCallRecord], operator.add]
    usage: Usage
    turns: int
    reminders: int
    limit_hit: StopReason | None
    submission: BaselineSubmission | None
    errors: Annotated[list[str], operator.add]
    stop_reason: StopReason | None
    report: InvestigationReport | None


def init_baseline_state(incident: dict, config: InvestigationConfig | None = None, now: float | None = None) -> BaselineState:
    return BaselineState(incident=incident, config=config or InvestigationConfig(),
                         started_at=time.time() if now is None else now, messages=[], tool_records=[],
                         usage=Usage(), turns=0, reminders=0, limit_hit=None, submission=None, errors=[],
                         stop_reason=None, report=None)


def _limit(state: BaselineState, now: float) -> StopReason | None:
    limits = state["config"].limits
    usage = state["usage"]
    if len(state.get("tool_records", [])) >= limits.max_tool_calls:
        return "max_tool_calls"
    if usage.tokens >= limits.max_tokens:
        return "max_tokens"
    if now - state["started_at"] >= limits.max_wall_clock_s:
        return "max_wall_clock"
    return None


# --- conversion to the shared report -----------------------------------------------------------


def to_report(state: BaselineState) -> tuple[InvestigationReport, list[str]]:
    errors: list[str] = []
    records = {r.call_id: r for r in state.get("tool_records", [])}
    gaps = []
    for r in records.values():
        if not r.ok:
            gaps.append(f"{r.tool} {r.args} failed: {r.error_kind}: {r.error_message}")
        gaps.extend(f"{r.tool} {r.args}: {note}" for note in r.quality_notes)
    sub = state.get("submission")
    stop = state.get("stop_reason")
    title = str(state["incident"].get("title", "incident"))
    if sub is None:
        return InvestigationReport(incident_title=title, outcome="inconclusive", confidence="low",
                                   summary="The baseline agent did not submit a report.", hypotheses=[],
                                   corrective_actions=[], gaps=gaps, stop_reason=stop), errors

    evidence: list[EvidenceIndexEntry] = []
    ids: dict[tuple[frozenset, str], str] = {}

    def cite(fact: BaselineFact) -> str | None:
        """Map a cited fact to an evidence id; facts citing no real tool call are dropped."""
        real = [c for c in fact.call_ids if c in records]
        if len(real) < len(fact.call_ids):
            errors.append(f"baseline cited unknown call ids {sorted(set(fact.call_ids) - set(real))}: {fact.text[:60]!r}")
        if not real:
            return None
        key = (frozenset(real), fact.text)
        if key not in ids:
            ids[key] = f"E{len(evidence) + 1}"
            evidence.append(EvidenceIndexEntry(id=ids[key], summary=fact.text,
                                               tools=sorted({records[c].tool for c in real}), source_call_ids=real))
        return ids[key]

    hypotheses = []
    for n, h in enumerate(sorted(sub.hypotheses, key=lambda h: h.rank), start=1):
        sup = [e for e in (cite(f) for f in h.supporting) if e]
        con = [e for e in (cite(f) for f in h.contradicting) if e]
        facts = [Claim(text=f.text, evidence_ids=[e]) for f, e in
                 [(f, cite(f)) for f in h.supporting + h.contradicting] if e]
        hypotheses.append(ReportHypothesis(rank=n, hypothesis_id=f"H{n}", statement=h.statement, status=h.status,
                                           confidence=h.confidence, supporting_evidence=sup,
                                           contradicting_evidence=con, facts=facts, assumptions=h.assumptions))
    actions = []
    for n, a in enumerate(sorted(sub.corrective_actions, key=lambda a: a.rank), start=1):
        ids = [e.id for e in evidence if set(e.source_call_ids) & set(a.call_ids)]
        actions.append(CorrectiveActionOut(rank=n, category=a.category, action=a.action, rationale=a.rationale,
                                           evidence_ids=ids))
    top = hypotheses[0] if hypotheses else None
    # Same rule as the multi-agent report: claimed, submitted normally, and backed by a supported top hypothesis.
    conclusive = (sub.outcome == "conclusive" and stop == "accepted" and top is not None
                  and top.status == "supported" and top.confidence in ("high", "medium"))
    return InvestigationReport(
        incident_title=title,
        outcome="conclusive" if conclusive else "inconclusive",
        summary=sub.summary,
        root_cause=(sub.root_cause or (top.statement if top else None)) if conclusive else None,
        confidence=sub.confidence if conclusive else "low",
        hypotheses=hypotheses,
        corrective_actions=actions,
        missing_data=sub.missing_data,
        gaps=list(dict.fromkeys(gaps)),
        security_findings=sub.security_findings,
        questions_for_human=sub.questions_for_human,
        evidence_index=evidence,
        stop_reason=stop,
    ), errors


# --- graph -----------------------------------------------------------------------------------


def build_baseline_graph(model: BaselineModel | None = None, *, context_schema: type = RunContext,
                         bootstrap: Callable[[Any], dict] | None = None):
    """`model=None` resolves it from `runtime.context.baseline` (Studio)."""

    def _model(runtime) -> BaselineModel:
        resolved = model or getattr(runtime.context, "baseline", None)
        if resolved is None:
            raise RuntimeError("no baseline model: pass one to build_baseline_graph() or provide it in the context")
        return resolved

    def start(state: BaselineState, runtime: Runtime) -> dict:
        update = {} if state.get("incident") else bootstrap(runtime.context)
        incident = update.get("incident") or state["incident"]
        cfg = update.get("config") or state["config"]
        brief = (f"Incident report:\n{render_incident(incident)}\n\n"
                 f"Tool-call budget: {cfg.limits.max_tool_calls}. When done, call {SUBMIT_REPORT}.")
        return {**update, "messages": [SystemMessage(prompts.BASELINE), HumanMessage(brief)]}

    def agent(state: BaselineState, runtime: Runtime) -> dict:
        try:
            ai, usage = _model(runtime)(state["messages"])
        except Exception as exc:  # recovery: a failed model call ends the run with a fallback report
            return {"errors": [f"baseline model call failed: {type(exc).__name__}: {exc}"], "stop_reason": "no_progress",
                    "turns": state["turns"] + 1}
        return {"messages": [ai], "usage": state["usage"] + usage, "turns": state["turns"] + 1}

    def route_agent(state: BaselineState) -> str:
        if state.get("stop_reason"):
            return "finalize"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "nudge" if state["reminders"] < MAX_REMINDERS else "finalize"

    def nudge(state: BaselineState) -> dict:
        reason = f"Budget reached ({state['limit_hit']}). " if state.get("limit_hit") else ""
        return {"messages": [HumanMessage(f"{reason}Call {SUBMIT_REPORT} now with your conclusions.")],
                "reminders": state["reminders"] + 1}

    def tools(state: BaselineState, runtime: Runtime) -> dict:
        limits = state["config"].limits
        used = len(state.get("tool_records", []))
        toolbox = ToolBox(runtime.context.harness, max(0, limits.max_tool_calls - used))
        out: list[AnyMessage] = []
        submission = None
        for call in state["messages"][-1].tool_calls:
            if call["name"] == SUBMIT_REPORT:
                try:
                    submission = BaselineSubmission.model_validate(call["args"])
                    out.append(ToolMessage("received", tool_call_id=call["id"]))
                except ValidationError as exc:
                    out.append(ToolMessage(f"invalid report: {exc.errors()[:3]}; fix and call again",
                                           tool_call_id=call["id"], status="error"))
            elif call["name"] not in TOOL_SPECS:
                out.append(ToolMessage(f"unknown tool {call['name']!r}", tool_call_id=call["id"], status="error"))
            else:
                out.append(ToolMessage(render_tool_result(toolbox.call(call["name"], call["args"])),
                                       tool_call_id=call["id"]))
        update: dict = {"messages": out, "tool_records": toolbox.records()}
        if submission is not None:
            update["submission"] = submission
        return update

    def route_tools(state: BaselineState, runtime: Runtime) -> str:
        if state.get("submission") is not None:
            return "finalize"
        if state["turns"] >= state["config"].limits.max_tool_calls + MAX_REMINDERS + 3:
            return "finalize"
        if _limit(state, runtime.context.clock()):
            return "budget" if state["reminders"] < MAX_REMINDERS else "finalize"
        return "agent"

    def budget(state: BaselineState, runtime: Runtime) -> dict:
        hit = _limit(state, runtime.context.clock())
        return {"limit_hit": hit, **nudge({**state, "limit_hit": hit})}

    def finalize(state: BaselineState) -> dict:
        stop = state.get("stop_reason")
        if stop is None:
            if state.get("submission") is not None:
                stop = state.get("limit_hit") or ("accepted" if state["submission"].outcome == "conclusive" else "inconclusive")
            else:
                stop = state.get("limit_hit") or "no_progress"
        report, errors = to_report({**state, "stop_reason": stop})
        return {"report": report, "stop_reason": stop, "errors": errors}

    g = StateGraph(BaselineState, context_schema=context_schema)
    for name, fn in [("start", start), ("agent", agent), ("tools", tools), ("nudge", nudge),
                     ("budget", budget), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "start")
    g.add_edge("start", "agent")
    g.add_conditional_edges("agent", route_agent, ["tools", "nudge", "finalize"])
    g.add_conditional_edges("tools", route_tools, ["agent", "budget", "finalize"])
    g.add_edge("nudge", "agent")
    g.add_edge("budget", "agent")
    g.add_edge("finalize", END)
    return g.compile()


# --- runner ------------------------------------------------------------------------------------


def run_baseline(
    scenario: Scenario,
    model: BaselineModel,
    config: InvestigationConfig | None = None,
    *,
    ledger: UsageLedger | None = None,
    tool_settings: RetrySettings | None = None,
    clock: Callable[[], float] = time.time,
) -> RunResult:
    config = config or InvestigationConfig()
    harness = ToolHarness(scenario, settings=tool_settings)
    started = clock()
    try:
        graph = build_baseline_graph(model)
        state = graph.invoke(
            init_baseline_state(scenario.incident_report.model_dump(mode="json"), config, now=started),
            {"recursion_limit": 3 * (config.limits.max_tool_calls + MAX_REMINDERS) + 20,
             "configurable": {"thread_id": str(uuid.uuid4())},
             "run_name": f"baseline:{scenario.id}", "tags": ["baseline", scenario.id],
             "metadata": {"scenario": scenario.id, "system": "baseline"}},
            context=RunContext(harness=harness, clock=clock, ledger=ledger),
        )
    finally:
        harness.close()
    usage = state["usage"].model_copy(update={"tool_calls": len(state["tool_records"])})
    return RunResult(
        state=state,
        report=state["report"],
        interrupted=False,
        usage=usage,
        tool_calls=len(harness.calls),
        latency_s=round(clock() - started, 3),
        by_agent={"baseline": usage},
        by_model=ledger.by_model() if ledger else {},
        calls=harness.calls,
    )
