"""Which model and effort each role uses.

A preset assigns each role a size tier (small / medium / large); the provider decides which
model fills each tier. Switching provider is one setting: INVESTIGATOR_PROVIDER=anthropic|openai
(or `--provider` on the CLIs).

Presets:
- budget   (default): every role one tier down from `balanced`, for cheap testing.
- balanced: large for reasoning/critic/baseline, medium for gatherers/planner/reporter, small for dependency.
- quality:  large everywhere.

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
Provider = Literal["anthropic", "openai"]
Tier = Literal["small", "medium", "large"]
TIERS: tuple[Tier, ...] = ("small", "medium", "large")

PROVIDER_MODELS: dict[Provider, dict[Tier, str]] = {
    "anthropic": {"small": "claude-haiku-5-5", "medium": "claude-sonnet-5-5", "large": "claude-opus-5-5"},
    # A general model for tool loops and prose, a reasoning model for judgment roles.
    # No larger reasoning tier is configured, so large reuses o3.
    "openai": {"small": "gpt-5.4", "medium": "o3", "large": "o3"},
}
DEFAULT_PROVIDER: Provider = "anthropic"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
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


PRESETS: dict[str, dict[Role, tuple[Tier, Effort]]] = {
    "balanced": {
        "triage": ("medium", "medium"),
        "observability": ("medium", "medium"),
        "change_history": ("medium", "medium"),
        "dependency": ("small", "low"),
        "reasoning": ("large", "high"),
        "critic": ("large", "high"),  # switch to another provider via MODEL_CRITIC for cross-family critique
        "reporter": ("medium", "medium"),
        "baseline": ("large", "high"),
    },
    "budget": {
        "triage": ("small", "medium"),
        "observability": ("small", "medium"),
        "change_history": ("small", "medium"),
        "dependency": ("small", "low"),
        "reasoning": ("medium", "high"),
        "critic": ("medium", "high"),
        "reporter": ("small", "medium"),
        "baseline": ("medium", "high"),
    },
    "quality": {role: ("large", "high") for role in ROLES},
}


def resolve_provider(provider: str | None = None, env: dict[str, str] | None = None) -> Provider:
    env = dict(os.environ if env is None else env)
    name = (provider or env.get("INVESTIGATOR_PROVIDER") or DEFAULT_PROVIDER).lower()
    if name not in PROVIDER_MODELS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDER_MODELS)}")
    return name  # type: ignore[return-value]


def load_model_config(
    preset: str | None = None, env: dict[str, str] | None = None, provider: str | None = None
) -> dict[Role, ModelSpec]:
    env = dict(os.environ if env is None else env)
    name = preset or env.get("INVESTIGATOR_PRESET", "budget")
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    chosen = resolve_provider(provider, env)
    models = PROVIDER_MODELS[chosen]
    config: dict[Role, ModelSpec] = {}
    for role, (tier, effort) in PRESETS[name].items():
        base = ModelSpec(provider=chosen, model=models[tier], effort=effort)
        ref = env.get(f"MODEL_{role.upper()}", base.ref)
        role_effort = env.get(f"EFFORT_{role.upper()}", base.effort)
        if ref != base.ref or role_effort != base.effort:
            base = ModelSpec.parse(ref, role_effort, max_tokens=base.max_tokens)  # type: ignore[arg-type]
        config[role] = base
    return config
