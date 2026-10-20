"""USD prices per million tokens, used to cost every LLM call.

Snapshots of first-party list prices: Anthropic (2026-10-06); OpenAI standard tier from
developers.openai.com/api/docs/models/<model> (2026-10-19). Verify before quoting numbers
externally; unknown models are counted as `unpriced_calls`, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache writes
    # Optional long-prompt tier: (threshold_input_tokens, input, output)
    long_prompt: tuple[int, float, float] | None = None


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    # Haiku 5.5: $0.10/$0.50 up to 100K prompt tokens, $0.50/$2.50 beyond. Cache rates assumed
    # at the standard 0.1x read / 1.25x write multipliers (not separately published in our source).
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125,
                              long_prompt=(100_000, 0.50, 2.50)),
    # OpenAI has no cache-write charge, so writes are priced as plain input.
    # gpt-5.4: prompts over 272K input tokens cost 2x input and 1.5x output.
    "gpt-5.4": Price(input=2.50, output=15.00, cache_read=0.25, cache_write=2.50,
                     long_prompt=(272_000, 5.00, 22.50)),
    "gpt-5.4-mini": Price(input=0.75, output=4.50, cache_read=0.075, cache_write=0.75),
    "o3": Price(input=2.00, output=8.00, cache_read=0.50, cache_write=2.00),
    "o4-mini": Price(input=1.10, output=4.40, cache_read=0.275, cache_write=1.10),
    # Standard tier, short context, from developers.openai.com/api/docs/pricing (2026-10-20).
    # Their long-context rates (2x input, 1.5x output) apply past a threshold the page doesn't state here.
    "gpt-6-astra": Price(input=10.00, output=50.00, cache_read=1.00, cache_write=12.50),
    "gpt-6.1-sol": Price(input=2.00, output=10.00, cache_read=0.10, cache_write=2.50),
}

# Snapshot suffixes providers append to the model id they report, e.g. "o3-2025-04-16".
_DATED = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


def _lookup(model: str) -> Price | None:
    # Exact id or a dated snapshot of it; never a prefix, so "o3-pro" is not priced as "o3".
    return PRICES.get(model) or PRICES.get(_DATED.sub("", model))


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float | None:
    """`input_tokens` is the provider total including cache reads/writes."""
    price = _lookup(model)
    if price is None:
        return None
    rate_in, rate_out = price.input, price.output
    if price.long_prompt and input_tokens > price.long_prompt[0]:
        _, rate_in, rate_out = price.long_prompt
    uncached = max(0, input_tokens - cache_read - cache_write)
    total = (
        uncached * rate_in
        + cache_read * price.cache_read
        + cache_write * price.cache_write
        + output_tokens * rate_out
    )
    return round(total / 1_000_000, 6)
