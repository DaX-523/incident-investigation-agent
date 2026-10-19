"""Build and run either system on one scenario. Shared by the CLI and the comparison runner."""

from __future__ import annotations

import os
from typing import Any, Literal

from dataset.schema import Scenario
from investigator.runner import RunResult, investigate
from investigator.schemas import InvestigationConfig

System = Literal["multi_agent", "baseline"]
SYSTEMS: tuple[System, ...] = ("multi_agent", "baseline")


def resolve_preset(preset: str | None, stub: bool) -> str:
    return "stub" if stub else preset or os.environ.get("INVESTIGATOR_PRESET", "budget")


def model_labels(system: System, preset: str, provider: str | None = None) -> dict[str, str]:
    if preset == "stub":
        return {}
    from investigator.llm.config import load_model_config

    config = load_model_config(preset, provider=provider)
    roles = ["baseline"] if system == "baseline" else [r for r in config if r != "baseline"]
    return {role: f"{config[role].model}/{config[role].effort}" for role in roles}


def run_system(scenario: Scenario, system: System, preset: str, config: InvestigationConfig | None = None,
               provider: str | None = None) -> RunResult:
    """preset "stub" runs the deterministic offline workers; anything else is an LLM preset.

    `provider` (anthropic | openai) defaults to INVESTIGATOR_PROVIDER, then anthropic.
    """
    config = config or InvestigationConfig()
    if preset == "stub":
        from investigator.agents.stubs import StubBaselineModel, stub_workers

        if system == "baseline":
            from investigator.baseline import run_baseline

            return run_baseline(scenario, StubBaselineModel(), config)
        return investigate(scenario, stub_workers(), config)

    from investigator.llm.config import load_model_config
    from investigator.llm.metering import UsageLedger

    models = load_model_config(preset, provider=provider)
    ledger = UsageLedger()
    if system == "baseline":
        from investigator.agents.llm_agents import _Caller
        from investigator.baseline import LLMBaselineModel, run_baseline
        from investigator.llm.factory import build_chat_model

        spec = models["baseline"]
        model = LLMBaselineModel(_Caller("baseline", spec, build_chat_model(spec, cache=True), ledger))
        return run_baseline(scenario, model, config, ledger=ledger)

    from investigator.agents.llm_agents import llm_workers

    return investigate(scenario, llm_workers(models, ledger), config, ledger=ledger)


def result_to_dict(result: RunResult) -> dict[str, Any]:
    """Everything needed to print, save and later re-score a run without re-running it."""
    return {
        "report": result.report.model_dump(mode="json") if result.report else None,
        "stop_reason": result.state.get("stop_reason"),
        "usage": result.usage.model_dump(),
        "by_agent": {k: v.model_dump() for k, v in result.by_agent.items()},
        "by_model": {k: v.model_dump() for k, v in result.by_model.items()},
        "tool_calls": result.tool_calls,
        "latency_s": result.latency_s,
        "errors": result.state.get("errors", []),
        "calls": [{"call_id": c.call_id, "tool": c.tool, "args": c.args, "ok": c.ok,
                   "error": c.error.kind if c.error else None, "attempts": c.attempts} for c in result.calls],
    }
