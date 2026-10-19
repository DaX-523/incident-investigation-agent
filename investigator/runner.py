"""Run one investigation end to end and collect the run metrics."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.types import Command

from dataset.schema import Scenario
from investigator.graph import RunContext, Workers, build_graph
from investigator.llm.metering import UsageLedger
from investigator.orchestrator.manager import init_state
from investigator.report import InvestigationReport
from investigator.schemas import InvestigationConfig, Usage
from investigator.tools.harness import RetrySettings, ToolHarness
from investigator.tools.models import ToolResult


@dataclass
class RunResult:
    state: dict[str, Any]
    report: InvestigationReport | None
    interrupted: bool
    usage: Usage
    tool_calls: int
    latency_s: float
    by_agent: dict[str, Usage]
    by_model: dict[str, Usage]
    calls: list[ToolResult] = field(default_factory=list)  # harness log, for grounding checks


def recursion_limit(config: InvestigationConfig) -> int:
    # triage + per round (orchestrator, workers, dependent workers, reasoning, critic) with headroom.
    return 20 + 10 * (config.limits.max_critic_loops + 2)


class Investigation:
    """Holds the graph, harness and thread for one incident so interactive runs can resume."""

    def __init__(
        self,
        scenario: Scenario,
        workers: Workers,
        config: InvestigationConfig | None = None,
        *,
        checkpointer=None,
        tool_settings: RetrySettings | None = None,
        clock: Callable[[], float] = time.time,
        ledger: UsageLedger | None = None,
    ):
        self.config = config or InvestigationConfig()
        self.harness = ToolHarness(scenario, settings=tool_settings)
        self.graph = build_graph(workers, checkpointer=checkpointer)
        self.ledger = ledger
        self.context = RunContext(harness=self.harness, clock=clock, ledger=ledger)
        self.thread = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": recursion_limit(self.config),
            # LangSmith trace labels.
            "run_name": f"investigation:{scenario.id}",
            "tags": ["multi_agent", scenario.id],
            "metadata": {"scenario": scenario.id, "system": "multi_agent", "mode": self.config.mode},
        }
        self.incident = scenario.incident_report.model_dump(mode="json")
        self._clock = clock
        self._started = clock()

    def start(self) -> RunResult:
        state = init_state(self.incident, self.config, now=self._started)
        return self._result(self.graph.invoke(state, self.thread, context=self.context))

    def resume(self, answer: str) -> RunResult:
        return self._result(self.graph.invoke(Command(resume=answer), self.thread, context=self.context))

    def close(self) -> None:
        self.harness.close()

    def _result(self, state: dict[str, Any]) -> RunResult:
        budget = state["budget"]
        return RunResult(
            state=state,
            report=state.get("report"),
            interrupted="__interrupt__" in state,
            usage=budget.usage,
            tool_calls=len(self.harness.calls),
            latency_s=round(self._clock() - self._started, 3),
            by_agent=dict(budget.by_agent),
            by_model=self.ledger.by_model() if self.ledger else {},
            calls=self.harness.calls,
        )


def investigate(scenario: Scenario, workers: Workers, config: InvestigationConfig | None = None, **kwargs) -> RunResult:
    run = Investigation(scenario, workers, config, **kwargs)
    try:
        return run.start()
    finally:
        run.close()
