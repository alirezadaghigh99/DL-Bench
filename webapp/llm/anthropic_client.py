"""Anthropic client (direct API, not Bedrock)."""

from __future__ import annotations

import os
import time

from .base import Generation


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            from anthropic import Anthropic  # lazy import
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, model: str, prompt: str, *, temperature: float = 0.0) -> Generation:
        client = self._ensure()
        start = time.monotonic()
        r = client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        duration = time.monotonic() - start
        text = "".join(block.text for block in r.content if block.type == "text")
        return Generation(
            model=model,
            text=text,
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            duration_s=duration,
        )
