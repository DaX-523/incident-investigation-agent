"""LLM-backed workers: planner, three gatherers (tool loops), reasoner, critic, reporter.

Each call is metered into a shared UsageLedger (tokens + USD by role and model).
Failures raise; the graph's worker wrapper retries and degrades to a failed Outcome.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from investigator.agents import prompts
from investigator.graph import Workers
from investigator.llm.config import ModelSpec, Role
from investigator.llm.factory import build_chat_model
from investigator.llm.metering import UsageLedger
from investigator.orchestrator.context import WorkerContext
from investigator.orchestrator.manager import ranked_hypotheses
from investigator.report import ReportNarrative
from investigator.schemas import (
    GATHERERS,
    CriticVerdict,
    Evidence,
    EvidenceDraft,
    Gap,
    Hypothesis,
    HypothesisAssessment,
    InvestigationState,
    Outcome,
    Task,
    ToolCallRecord,
    TriagePlan,
    Usage,
)
from investigator.tools.models import ToolResult
from investigator.tools.registry import TOOL_SPECS
from investigator.tools.toolbox import ToolBox

TOOLSETS: dict[str, list[str]] = {
    "observability": ["query_metrics", "search_logs", "get_traces", "get_alerts", "get_infra_events", "get_external_report"],
    "dependency": ["get_topology", "query_metrics", "get_alerts", "get_traces"],
    "change_history": ["get_changes", "get_infra_events", "search_past_incidents", "search_runbooks", "get_external_report"],
}
SUBMIT_TOOL = "submit_findings"


def render_toolsets() -> str:
    """Which specialist can run which tools, so the planner and critic route work to one that can answer it."""
    return "\n".join(f"- {agent}: {', '.join(tools)}" for agent, tools in TOOLSETS.items())
MAX_TOOL_RESULT_CHARS = 8_000
MAX_SUBMIT_REMINDERS = 2


class LLMRefusal(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


class GathererFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[EvidenceDraft] = Field(description="Observations, each citing call_ids from this task.")
    gaps: list[str] = Field([], description="Things you could not check, and why.")


class ReasoningOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[HypothesisAssessment]


# --- rendering ------------------------------------------------------------------------


def _dump(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def render_tool_result(result: ToolResult, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    payload = result.model_dump(mode="json", exclude={"latency_ms"})
    text = _dump(payload)
    if len(text) > max_chars:
        text = text[:max_chars] + f'... [investigator truncated {len(text) - max_chars} chars to fit context; narrow the query]'
    return text


def render_incident(incident: dict) -> str:
    keys = ("title", "severity", "reported_at", "reporter", "description")
    return "\n".join(f"{k}: {incident.get(k)}" for k in keys if incident.get(k))


def render_hypotheses(hypotheses: list[Hypothesis]) -> str:
    lines = []
    for h in hypotheses:
        line = f"- {h.id} [{h.origin}; status={h.status}"
        if h.rank is not None:
            line += f"; rank={h.rank}; confidence={h.confidence}"
        line += f"] {h.statement}"
        if h.supporting or h.contradicting:
            line += f" (supporting={h.supporting}, contradicting={h.contradicting})"
        if h.rationale:
            line += f"\n    rationale: {h.rationale}"
        lines.append(line)
    return "\n".join(lines) or "(none)"


def render_evidence(evidence: list[Evidence], limit: int = 80) -> str:
    lines = []
    for e in evidence[-limit:]:
        links = ", ".join(f"{link.hypothesis_id}:{link.stance}" for link in e.links)
        line = f"- {e.id} ({e.agent}, {'/'.join(e.tools)}; calls {','.join(e.source_call_ids)}) {e.summary}"
        if links:
            line += f" [links: {links}]"
        if e.quality_notes:
            line += f" [data quality: {'; '.join(e.quality_notes)}]"
        lines.append(line)
    return "\n".join(lines) or "(no evidence yet)"


def render_gaps(gaps: list[Gap], limit: int = 30) -> str:
    return "\n".join(f"- [{g.kind}] {g.description}" for g in gaps[-limit:]) or "(none)"


def render_tool_calls(calls: list[ToolCallRecord], limit: int = 40) -> str:
    lines = []
    for c in calls[-limit:]:
        if not c.ok:
            result = f"failed ({c.error_kind})"
        else:
            result = "no results" if c.empty else "ok"
        lines.append(f"- {c.tool} {_dump(c.args)} -> {result}")
    return "\n".join(lines) or "(none)"


# --- model calls ----------------------------------------------------------------------


class _Caller:
    def __init__(self, role: Role, spec: ModelSpec, model: BaseChatModel, ledger: UsageLedger):
        self.role, self.spec, self.model, self.ledger = role, spec, model, ledger

    def _check(self, message: AIMessage) -> None:
        meta = message.response_metadata or {}
        if meta.get("stop_reason") == "refusal":  # Anthropic
            raise LLMRefusal(f"{self.role}: model declined the request ({meta.get('stop_details')})")
        if message.additional_kwargs.get("refusal") or meta.get("finish_reason") == "content_filter":  # OpenAI
            raise LLMRefusal(f"{self.role}: model declined the request ({message.additional_kwargs.get('refusal')})")

    def invoke(self, runnable, messages: list[BaseMessage]) -> tuple[AIMessage, Usage]:
        started = time.monotonic()
        message = runnable.invoke(messages)
        usage = self.ledger.record(self.role, message, self.spec.model, (time.monotonic() - started) * 1000)
        self._check(message)
        return message, usage

    def structured(self, schema: type[BaseModel], messages: list[BaseMessage]) -> tuple[Any, Usage]:
        runnable = self.model.with_structured_output(schema, method="json_schema", include_raw=True)
        started = time.monotonic()
        out = runnable.invoke(messages)
        raw: AIMessage = out["raw"]
        usage = self.ledger.record(self.role, raw, self.spec.model, (time.monotonic() - started) * 1000)
        self._check(raw)
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise StructuredOutputError(f"{self.role}: could not parse {schema.__name__}: {out.get('parsing_error')}")
        parsed = out["parsed"]
        return (parsed if isinstance(parsed, schema) else schema.model_validate(parsed)), usage


def _tool_def(name: str) -> dict:
    spec = TOOL_SPECS[name]
    return {"name": name, "description": spec.description, "input_schema": spec.input_model.model_json_schema()}


SUBMIT_DEF = {
    "name": SUBMIT_TOOL,
    "description": "Submit your findings when you are done investigating. Call exactly once, at the end.",
    "input_schema": GathererFindings.model_json_schema(),
}


# --- workers ----------------------------------------------------------------------------


class LLMPlanner:
    def __init__(self, caller: _Caller):
        self.caller = caller

    def __call__(self, incident: dict, toolbox: ToolBox) -> TriagePlan:
        topology = toolbox.call("get_topology", {})
        services = topology.data["services"] if topology.ok and topology.data else []
        topo = "\n".join(f"- {s['name']} ({s['kind']}) -> {', '.join(s['depends_on']) or '-'}" for s in services)
        # Recent changes across all services, so a change the reporter never mentioned can still become a hypothesis.
        changes = toolbox.call("get_changes", _sweep_args(incident.get("reported_at")))
        listed = "\n".join(f"- {c['ts']} {c['service']} [{c['kind']}] {c['summary']}"
                           for c in (changes.data or {}).get("changes", [])) if changes.ok else ""
        messages = [
            SystemMessage(prompts.PLANNER),
            HumanMessage(f"Incident report:\n{render_incident(incident)}\n\nService topology:\n{topo or '(unavailable)'}"
                         f"\n\nChanges in all services over the {CHANGE_SWEEP_DAYS} days before the report:\n"
                         f"{listed or '(none returned)' if changes.ok else '(change history unavailable)'}"
                         f"\n\nSpecialists and the tools each can use:\n{render_toolsets()}"),
        ]
        plan, _ = self.caller.structured(TriagePlan, messages)
        return plan


class LLMGatherer:
    def __init__(self, agent: str, caller: _Caller):
        self.agent = agent
        self.caller = caller
        self.tool_names = TOOLSETS[agent]
        self.bound = caller.model.bind_tools([_tool_def(n) for n in self.tool_names] + [SUBMIT_DEF])

    def _brief(self, task: Task, ctx: WorkerContext) -> str:
        triage = ctx.triage
        parts = [
            f"Incident report:\n{render_incident(ctx.incident)}",
            f"Triage summary: {triage.summary}" if triage else "",
            f"Incident window (planner estimate): {triage.window_start} .. {triage.window_end}" if triage and triage.window_start else "",
            f"YOUR TASK ({task.task_id}): {task.goal}",
            f"Focus services: {', '.join(task.services)}" if task.services else "",
            f"Hypotheses under test (link evidence to these ids):\n{render_hypotheses(ctx.hypotheses)}",
            f"Evidence already collected by the team (don't duplicate):\n{render_evidence(ctx.evidence, 30)}",
            f"Queries already run by the team (don't repeat them; their findings are in the evidence above):\n"
            f"{render_tool_calls(ctx.tool_calls)}" if ctx.tool_calls else "",
            f"Known gaps:\n{render_gaps(ctx.gaps, 15)}",
            "Human input:\n" + "\n".join(ctx.human_inputs) if ctx.human_inputs else "",
            f"Tool-call allowance for this task: {ctx.tool_call_allowance}. When done, call {SUBMIT_TOOL}.",
        ]
        return "\n\n".join(p for p in parts if p)

    def __call__(self, task: Task, ctx: WorkerContext, toolbox: ToolBox) -> Outcome:
        messages: list[BaseMessage] = [SystemMessage(prompts.gatherer_prompt(self.agent)), HumanMessage(self._brief(task, ctx))]
        if self.agent == "change_history" and not _has_change_sweep(ctx.tool_calls) and toolbox.remaining > 0:
            # Fixed first step, not left to the model: a service-filtered change query can miss the change
            # that matters (it is often in a neighbouring service), so one unfiltered sweep always runs.
            # (The planner saw a sweep too, but evidence can only cite this worker's own calls.)
            anchor = ctx.triage.window_start if ctx.triage and ctx.triage.window_start else ctx.incident.get("reported_at")
            sweep = toolbox.call("get_changes", _sweep_args(anchor))
            messages.append(HumanMessage(
                f"Before you start, an unfiltered change-history sweep (all services, from {CHANGE_SWEEP_DAYS} days "
                f"before the incident) was run for this task; cite it as {sweep.call_id}:\n{render_tool_result(sweep)}"))
        usage = Usage()
        findings: GathererFindings | None = None
        reminders = 0
        max_turns = ctx.tool_call_allowance + MAX_SUBMIT_REMINDERS + 2
        for _ in range(max_turns):
            ai, u = self.caller.invoke(self.bound, messages)
            usage = usage + u
            messages.append(ai)
            if not ai.tool_calls:
                if reminders >= MAX_SUBMIT_REMINDERS:
                    break
                reminders += 1
                messages.append(HumanMessage(f"{prompts.GATHERER_REPORT}\nCall {SUBMIT_TOOL} now."))
                continue
            for call in ai.tool_calls:
                if call["name"] == SUBMIT_TOOL:
                    try:
                        findings = GathererFindings.model_validate(call["args"])
                        messages.append(ToolMessage("received", tool_call_id=call["id"]))
                    except ValidationError as exc:
                        messages.append(ToolMessage(f"invalid findings: {exc.errors()[:3]}; fix and call again",
                                                    tool_call_id=call["id"], status="error"))
                elif call["name"] not in self.tool_names:
                    messages.append(ToolMessage(f"tool {call['name']!r} is not available to you; use {self.tool_names}",
                                                tool_call_id=call["id"], status="error"))
                else:
                    result = toolbox.call(call["name"], call["args"])
                    messages.append(ToolMessage(render_tool_result(result), tool_call_id=call["id"]))
            if findings is not None:
                break
            if toolbox.remaining <= 0:
                messages.append(HumanMessage(f"Tool allowance used up. {prompts.GATHERER_REPORT}\nCall {SUBMIT_TOOL} now."))
        if findings is None:
            raise StructuredOutputError(f"{self.agent}: no findings submitted")
        status = "ok" if not findings.gaps else "partial"
        return Outcome(task_id=task.task_id, agent=task.agent, status=status,
                       evidence=findings.evidence, gaps=findings.gaps, usage=usage)


CHANGE_SWEEP_DAYS = 14


def _has_change_sweep(calls: list[ToolCallRecord]) -> bool:
    return any(c.tool == "get_changes" and c.ok and not c.args.get("service") and not c.args.get("kind") for c in calls)


def _sweep_args(anchor: datetime | str | None) -> dict[str, str]:
    """get_changes args covering CHANGE_SWEEP_DAYS before `anchor`; unbounded history if it is unknown."""
    if isinstance(anchor, str):
        try:
            anchor = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
        except ValueError:
            anchor = None
    if not isinstance(anchor, datetime):
        return {}  # unbounded history
    start = anchor - timedelta(days=CHANGE_SWEEP_DAYS)
    return {"start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")}


def _analysis_brief(ctx: WorkerContext) -> str:
    verdict = ctx.last_verdict
    parts = [
        f"Incident report:\n{render_incident(ctx.incident)}",
        f"Triage summary: {ctx.triage.summary}" if ctx.triage else "",
        f"Hypotheses:\n{render_hypotheses(ctx.hypotheses)}",
        f"Evidence:\n{render_evidence(ctx.evidence)}",
        f"Gaps (failed tools, data-quality problems, unchecked areas):\n{render_gaps(ctx.gaps)}",
        f"Previous critic verdict: {verdict.verdict}; issues: {verdict.issues}" if verdict else "",
        "Human input:\n" + "\n".join(ctx.human_inputs) if ctx.human_inputs else "",
    ]
    return "\n\n".join(p for p in parts if p)


class LLMReasoner:
    def __init__(self, caller: _Caller):
        self.caller = caller

    def __call__(self, task: Task, ctx: WorkerContext) -> Outcome:
        out, usage = self.caller.structured(ReasoningOutput, [SystemMessage(prompts.REASONER), HumanMessage(_analysis_brief(ctx))])
        return Outcome(task_id=task.task_id, agent="reasoning", status="ok", assessments=out.assessments, usage=usage)


class LLMCritic:
    def __init__(self, caller: _Caller):
        self.caller = caller

    def __call__(self, task: Task, ctx: WorkerContext) -> Outcome:
        final = "accept, inconclusive or needs_human" if ctx.human_available else "accept or inconclusive"
        brief = _analysis_brief(ctx) + f"\n\nSpecialists and the tools each can use:\n{render_toolsets()}" + (
            f"\n\nFurther evidence rounds available after this one: {ctx.critic_rounds_left}. "
            f"If none remain, choose {final}."
        )
        brief += f"\n{ctx.budget_note}" if ctx.budget_note else ""
        if ctx.final_call:
            brief += (f"\nThe investigation budget is spent ({ctx.final_call}); no more evidence can be gathered. "
                      "Decide now from what you have: accept if the evidence supports the top hypothesis (at the "
                      "confidence it allows), otherwise inconclusive with the blocking gaps.")
        if not ctx.human_available:
            # In eval mode needs_human can only end the run as inconclusive, whatever the evidence says.
            brief += ("\nNo human is available in this run, so never choose needs_human. Decide from the "
                      "evidence: put questions a human could still answer in questions_for_human.")
        verdict, usage = self.caller.structured(CriticVerdict, [SystemMessage(prompts.CRITIC), HumanMessage(brief)])
        return Outcome(task_id=task.task_id, agent="critic", status="ok", verdict=verdict, usage=usage)


class LLMReporter:
    def __init__(self, caller: _Caller):
        self.caller = caller

    def __call__(self, state: InvestigationState) -> ReportNarrative:
        verdict = state["verdicts"][-1] if state.get("verdicts") else None
        brief = "\n\n".join(p for p in [
            f"Incident report:\n{render_incident(state['incident'])}",
            f"Investigation stopped because: {state.get('stop_reason')}",
            f"Ranked hypotheses:\n{render_hypotheses(ranked_hypotheses(state))}",
            f"Evidence:\n{render_evidence(state['evidence'])}",
            f"Gaps:\n{render_gaps(state['gaps'])}",
            f"Last critic verdict: {verdict.verdict}; issues: {verdict.issues}" if verdict else "",
        ] if p)
        narrative, _ = self.caller.structured(ReportNarrative, [SystemMessage(prompts.REPORTER), HumanMessage(brief)])
        return narrative


def llm_workers(config: dict[Role, ModelSpec], ledger: UsageLedger, models: dict[Role, BaseChatModel] | None = None) -> Workers:
    """Build all workers. `models` lets tests inject fake chat models per role."""
    def caller(role: Role) -> _Caller:
        model = (models or {}).get(role) or build_chat_model(config[role], cache=role in GATHERERS)
        return _Caller(role, config[role], model, ledger)

    return Workers(
        planner=LLMPlanner(caller("triage")),
        gatherers={a: LLMGatherer(a, caller(a)) for a in GATHERERS},
        reasoner=LLMReasoner(caller("reasoning")),
        critic=LLMCritic(caller("critic")),
        reporter=LLMReporter(caller("reporter")),
    )
