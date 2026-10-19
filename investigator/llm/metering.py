"""Per-call token and cost ledger, broken down by role and by model."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from investigator.llm.pricing import cost_usd
from investigator.schemas import Usage


@dataclass(frozen=True)
class LLMCall:
    role: str
    model: str
    usage: Usage


def usage_from_message(message: AIMessage, fallback_model: str, latency_ms: float = 0.0) -> tuple[str, Usage]:
    """Read LangChain usage metadata; price it on the model that actually answered."""
    meta = message.usage_metadata or {}
    details = meta.get("input_token_details") or {}
    out_details = meta.get("output_token_details") or {}
    response_meta = message.response_metadata or {}
    model = str(response_meta.get("model_name") or response_meta.get("model") or fallback_model)

    input_tokens = int(meta.get("input_tokens") or 0)
    output_tokens = int(meta.get("output_tokens") or 0)
    cache_read = int(details.get("cache_read") or 0)
    cache_write = int(details.get("cache_creation") or 0) + int(details.get("ephemeral_5m_input_tokens") or 0) \
        + int(details.get("ephemeral_1h_input_tokens") or 0)
    cost = cost_usd(model, input_tokens, output_tokens, cache_read, cache_write)
    return model, Usage(
        llm_calls=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=int(out_details.get("reasoning") or 0),
        cost_usd=cost or 0.0,
        unpriced_calls=0 if cost is not None else 1,
        latency_ms=round(latency_ms, 2),
    )


class UsageLedger:
    """Thread-safe: parallel gatherers record into the same ledger."""

    def __init__(self) -> None:
        self._calls: list[LLMCall] = []
        self._lock = threading.Lock()

    def record(self, role: str, message: AIMessage, fallback_model: str, latency_ms: float = 0.0) -> Usage:
        model, usage = usage_from_message(message, fallback_model, latency_ms)
        with self._lock:
            self._calls.append(LLMCall(role=role, model=model, usage=usage))
        return usage

    @property
    def calls(self) -> list[LLMCall]:
        with self._lock:
            return list(self._calls)

    def total(self, role: str | None = None) -> Usage:
        total = Usage()
        for c in self.calls:
            if role is None or c.role == role:
                total = total + c.usage
        return total

    def by_role(self) -> dict[str, Usage]:
        return self._group(lambda c: c.role)

    def by_model(self) -> dict[str, Usage]:
        return self._group(lambda c: c.model)

    def _group(self, key) -> dict[str, Usage]:
        groups: dict[str, Usage] = {}
        for c in self.calls:
            groups[key(c)] = groups.get(key(c), Usage()) + c.usage
        return groups
