"""LangGraph Studio / `langgraph dev` entry point.

The run context is plain JSON (scenario, mode, preset); harness, ledger and workers are built
per run from it. Start with: `uv run langgraph dev` (see langgraph.json).
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from investigator.graph import Workers, build_graph
from investigator.llm.metering import UsageLedger
from investigator.orchestrator.manager import init_state
from investigator.scenario import load_scenario, scenario_ids
from investigator.schemas import InvestigationConfig
from investigator.tools.harness import ToolHarness

ScenarioId = Literal[tuple(scenario_ids())]  # type: ignore[valid-type]


class StudioContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: ScenarioId = Field(default=scenario_ids()[0], description="Incident scenario to investigate.")  # type: ignore[valid-type]
    mode: Literal["stub", "llm"] = Field("stub", description="stub = deterministic, no LLM calls; llm = real models.")
    preset: Literal["budget", "balanced", "quality"] = Field("budget", description="Model preset when mode=llm.")
    investigation_mode: Literal["interactive", "eval"] = Field(
        "interactive", description="interactive pauses for human review when the critic asks; eval never pauses."
    )

    _harness: ToolHarness = PrivateAttr()
    _ledger: UsageLedger | None = PrivateAttr(default=None)
    _workers: Workers = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self._harness = ToolHarness(load_scenario(self.scenario))
        if self.mode == "llm":
            from investigator.agents.llm_agents import llm_workers
            from investigator.llm.config import load_model_config

            self._ledger = UsageLedger()
            self._workers = llm_workers(load_model_config(self.preset), self._ledger)
        else:
            from investigator.agents.stubs import stub_workers

            self._workers = stub_workers()

    @property
    def harness(self) -> ToolHarness:
        return self._harness

    @property
    def ledger(self) -> UsageLedger | None:
        return self._ledger

    @property
    def workers(self) -> Workers:
        return self._workers

    def clock(self) -> float:
        return time.time()


def _bootstrap(ctx: StudioContext) -> dict:
    scenario = load_scenario(ctx.scenario)
    return init_state(scenario.incident_report.model_dump(mode="json"),
                      InvestigationConfig(mode=ctx.investigation_mode), now=time.time())


graph = build_graph(context_schema=StudioContext, bootstrap=_bootstrap)
