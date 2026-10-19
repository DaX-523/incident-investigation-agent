"""USD prices per million tokens, used to cost every LLM call.

Snapshot of Anthropic first-party list prices (2026-10-06). Verify before quoting
numbers externally; unknown models are counted as `unpriced_calls`, never guessed.
"""

from __future__ import annotations

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
    # OpenAI models are added once the key and current price list are confirmed.
}


def _lookup(model: str) -> Price | None:
    if model in PRICES:
        return PRICES[model]
    for name, price in PRICES.items():  # tolerate dated/suffixed ids
        if model.startswith(name):
            return price
    return None


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
