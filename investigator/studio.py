"""LangGraph Studio / `langgraph dev` entry point.

The run context is plain JSON (scenario, mode, preset); harness, ledger and workers are built
per run from it. Start with: `uv run langgraph dev` (see langgraph.json).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from investigator.baseline import build_baseline_graph
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
    provider: Literal["anthropic", "openai"] = Field("anthropic", description="Model provider when mode=llm.")
    investigation_mode: Literal["interactive", "eval"] = Field(
        "interactive", description="interactive pauses for human review when the critic asks; eval never pauses."
    )

    _harness: ToolHarness = PrivateAttr()
    _ledger: UsageLedger | None = PrivateAttr(default=None)
    _workers: Workers = PrivateAttr()
    _baseline: Any = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self._harness = ToolHarness(load_scenario(self.scenario))
        if self.mode == "llm":
            from investigator.agents.llm_agents import _Caller, llm_workers
            from investigator.baseline import LLMBaselineModel
            from investigator.llm.config import load_model_config
            from investigator.llm.factory import build_chat_model

            config = load_model_config(self.preset, provider=self.provider)
            self._ledger = UsageLedger()
            self._workers = llm_workers(config, self._ledger)
            spec = config["baseline"]
            self._baseline = LLMBaselineModel(_Caller("baseline", spec, build_chat_model(spec, cache=True), self._ledger))
        else:
            from investigator.agents.stubs import StubBaselineModel, stub_workers

            self._workers = stub_workers()
            self._baseline = StubBaselineModel()

    @property
    def harness(self) -> ToolHarness:
        return self._harness

    @property
    def ledger(self) -> UsageLedger | None:
        return self._ledger

    @property
    def workers(self) -> Workers:
        return self._workers

    @property
    def baseline(self):
        return self._baseline

    def clock(self) -> float:
        return time.time()


def _bootstrap(ctx: StudioContext) -> dict:
    scenario = load_scenario(ctx.scenario)
    return init_state(scenario.incident_report.model_dump(mode="json"),
                      InvestigationConfig(mode=ctx.investigation_mode), now=time.time())


def _bootstrap_baseline(ctx: StudioContext) -> dict:
    from investigator.baseline import init_baseline_state

    scenario = load_scenario(ctx.scenario)
    return init_baseline_state(scenario.incident_report.model_dump(mode="json"),
                               InvestigationConfig(mode="eval"), now=time.time())


graph = build_graph(context_schema=StudioContext, bootstrap=_bootstrap)
baseline_graph = build_baseline_graph(context_schema=StudioContext, bootstrap=_bootstrap_baseline)
