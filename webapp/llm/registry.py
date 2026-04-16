"""Map a model id to the client that owns it.

Adding a model: put its price in pricing.PRICES and add a row below.
Adding a provider: write a client implementing LLMClient and register it here.
"""

from __future__ import annotations

from .anthropic_client import AnthropicClient
from .base import LLMClient
from .openai_client import OpenAIClient


_OPENAI = OpenAIClient()
_ANTHROPIC = AnthropicClient()

# model_id -> client
MODELS: dict[str, LLMClient] = {
    "gpt-4o":              _OPENAI,
    "gpt-4o-mini":         _OPENAI,
    "o3-mini":             _OPENAI,
    "claude-opus-4-7":     _ANTHROPIC,
    "claude-sonnet-4-6":   _ANTHROPIC,
    "claude-haiku-4-5":    _ANTHROPIC,
}


def client_for(model: str) -> LLMClient:
    if model not in MODELS:
        raise KeyError(f"unknown model: {model!r}. Known: {sorted(MODELS)}")
    return MODELS[model]


def known_models() -> list[str]:
    return sorted(MODELS)
