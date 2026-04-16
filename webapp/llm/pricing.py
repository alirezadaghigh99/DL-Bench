"""Per-model price table, USD per 1M tokens (input / output).

Numbers are list prices at time of writing (April 2026). Update freely; the
cache stores the cost that was *recorded at generation time*, so edits here
won't retroactively change historical totals.

If a model isn't in the table the cost computes to 0.0 (so unknown providers
don't crash the UI) and the entry is logged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


PRICES: dict[str, Price] = {
    # OpenAI
    "gpt-4o":             Price(2.50,  10.00),
    "gpt-4o-mini":        Price(0.15,   0.60),
    "o3-mini":            Price(1.10,   4.40),
    # Anthropic
    "claude-opus-4-7":    Price(15.00, 75.00),
    "claude-sonnet-4-6":  Price( 3.00, 15.00),
    "claude-haiku-4-5":   Price( 0.80,  4.00),
    "claude-3-5-sonnet-20240620": Price(3.00, 15.00),
    # Fireworks (approximate)
    "deepseek-v3":        Price(0.27,  1.10),
    "llama-v3p1-70b-instruct": Price(0.90, 0.90),
    "mixtral-8x22b-instruct":  Price(0.90, 0.90),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    return (input_tokens * p.input_per_m + output_tokens * p.output_per_m) / 1_000_000


def known() -> list[str]:
    return sorted(PRICES)
