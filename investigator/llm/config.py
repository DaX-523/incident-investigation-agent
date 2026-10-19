"""Which model and effort each role uses.

Presets:
- budget   (default): every role one tier down from `balanced`, for cheap testing.
- balanced: Opus 5.5 for reasoning/critic/baseline, Sonnet 5.5 for gatherers/planner/reporter, Haiku 5.5 for dependency.
- quality:  Opus 5.5 everywhere.

Override per role with env: MODEL_<ROLE>=<provider>:<model>, EFFORT_<ROLE>=low|medium|high|xhigh|max.
Select the preset with INVESTIGATOR_PRESET.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["triage", "observability", "dependency", "change_history", "reasoning", "critic", "reporter", "baseline"]
ROLES: tuple[Role, ...] = ("triage", "observability", "dependency", "change_history", "reasoning", "critic", "reporter", "baseline")
Effort = Literal["low", "medium", "high", "xhigh", "max"]

OPUS = "anthropic:claude-opus-5-5"
SONNET = "anthropic:claude-sonnet-5-5"
HAIKU = "anthropic:claude-haiku-5-5"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["anthropic", "openai"]
    model: str
    effort: Effort
    max_tokens: int = 16_000

    @classmethod
    def parse(cls, ref: str, effort: Effort, **kw) -> ModelSpec:
        provider, _, model = ref.partition(":")
        if not model:
            raise ValueError(f"model reference must look like 'provider:model', got {ref!r}")
        return cls(provider=provider, model=model, effort=effort, **kw)

    @property
    def ref(self) -> str:
        return f"{self.provider}:{self.model}"


def _preset(table: dict[Role, tuple[str, Effort]]) -> dict[Role, ModelSpec]:
    return {role: ModelSpec.parse(ref, effort) for role, (ref, effort) in table.items()}


PRESETS: dict[str, dict[Role, ModelSpec]] = {
    "balanced": _preset({
        "triage": (SONNET, "medium"),
        "observability": (SONNET, "medium"),
        "change_history": (SONNET, "medium"),
        "dependency": (HAIKU, "low"),
        "reasoning": (OPUS, "high"),
        "critic": (OPUS, "high"),  # switch to an OpenAI model via MODEL_CRITIC for cross-family critique
        "reporter": (SONNET, "medium"),
        "baseline": (OPUS, "high"),
    }),
    "budget": _preset({
        "triage": (HAIKU, "medium"),
        "observability": (HAIKU, "medium"),
        "change_history": (HAIKU, "medium"),
        "dependency": (HAIKU, "low"),
        "reasoning": (SONNET, "high"),
        "critic": (SONNET, "high"),
        "reporter": (HAIKU, "medium"),
        "baseline": (SONNET, "high"),
    }),
    "quality": _preset({role: (OPUS, "high") for role in ROLES}),
}


def load_model_config(preset: str | None = None, env: dict[str, str] | None = None) -> dict[Role, ModelSpec]:
    env = dict(os.environ if env is None else env)
    name = preset or env.get("INVESTIGATOR_PRESET", "budget")
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    config = dict(PRESETS[name])
    for role in ROLES:
        base = config[role]
        ref = env.get(f"MODEL_{role.upper()}", base.ref)
        effort = env.get(f"EFFORT_{role.upper()}", base.effort)
        if ref != base.ref or effort != base.effort:
            config[role] = ModelSpec.parse(ref, effort, max_tokens=base.max_tokens)  # type: ignore[arg-type]
    return config
