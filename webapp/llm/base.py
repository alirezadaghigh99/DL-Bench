"""Provider-agnostic LLM client interface.

A `Generation` carries everything we need to bill, cache, and display: the
text, the token usage as reported by the provider (so we record exactly what
they billed us for), the wall time, and the model id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .pricing import cost_usd


@dataclass(frozen=True)
class Generation:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    duration_s: float

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


class LLMClient(Protocol):
    name: str

    def generate(self, model: str, prompt: str, *, temperature: float = 0.0) -> Generation: ...
