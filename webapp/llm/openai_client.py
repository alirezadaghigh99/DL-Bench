"""OpenAI client. Lazy-imports the SDK so the webapp boots even if openai
isn't installed (e.g. you're only using Anthropic models)."""

from __future__ import annotations

import os
import time

from .base import Generation


class OpenAIClient:
    name = "openai"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            from openai import OpenAI  # lazy import
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, model: str, prompt: str, *, temperature: float = 0.0) -> Generation:
        client = self._ensure()
        start = time.monotonic()
        kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        # o-series models reject temperature, max_tokens, etc.
        if not model.startswith("o"):
            kwargs["temperature"] = temperature
        r = client.chat.completions.create(**kwargs)
        duration = time.monotonic() - start
        usage = r.usage
        return Generation(
            model=model,
            text=r.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            duration_s=duration,
        )
