"""Builds LangChain chat models from a ModelSpec."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from investigator.llm.config import ModelSpec

# Server-side refusal fallback: supported on these Claude models (not Haiku 5.5).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_chat_model(spec: ModelSpec, *, cache: bool = False, timeout_s: float = 120.0, max_retries: int = 2) -> BaseChatModel:
    """`cache=True` for multi-turn tool loops, which resend a growing prefix every turn. One-shot calls
    would only pay the cache-write premium."""
    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_kwargs: dict = {"cache_control": {"type": "ephemeral"}} if cache else {}
        kwargs: dict = {}
        if spec.model in _FALLBACK_MODELS:
            # Security-heavy telemetry (injection payloads, exploit strings) can trip safety
            # classifiers; route declines to a fallback model instead of failing the step.
            model_kwargs["fallbacks"] = "default"
            kwargs = {"betas": [_FALLBACK_BETA]}
        return ChatAnthropic(
            model=spec.model,
            model_kwargs=model_kwargs,
            effort=spec.effort,
            max_tokens=spec.max_tokens,
            max_retries=max_retries,
            default_request_timeout=timeout_s,
            **kwargs,
        )
    if spec.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI roles need `uv add langchain-openai` and OPENAI_API_KEY") from exc
        return ChatOpenAI(model=spec.model, reasoning_effort=spec.effort, max_retries=max_retries, timeout=timeout_s)
    raise ValueError(f"unsupported provider {spec.provider!r}")
