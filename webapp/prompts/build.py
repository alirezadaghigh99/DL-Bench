"""Prompting techniques.

Lifted to a clean shape. The legacy `src/prompts/` modules tangled prompt
construction with file I/O against a shots dataset; we keep that out of the
MVP and ship the two techniques that don't need extra data.

Adding `fewshot` later: write a `build_fewshot(prompt, shots)` here and add a
selector for shot examples in webapp/tasks.py.
"""

from __future__ import annotations

CODE_FORMAT_REMINDER = (
    "\n\nReturn the code in a single ```python ... ``` block. "
    "No explanation, no example usage."
)


def _zeroshot(prompt: str) -> str:
    return prompt + CODE_FORMAT_REMINDER


def _zeroshot_cot(prompt: str) -> str:
    return (
        prompt
        + "\n\nLet's generate the code step by step, reasoning about edge cases "
          "(shape mismatches, value errors, numerical stability) before writing it."
        + CODE_FORMAT_REMINDER
    )


def _zeroshot_guided(prompt: str) -> str:
    return (
        prompt
        + "\n\nBe aware of common bugs: shape mismatch, value errors, index "
          "out of range, and numerical errors (NaN, divide by zero)."
        + CODE_FORMAT_REMINDER
    )


TECHNIQUES = {
    "zeroshot":         _zeroshot,
    "zeroshot-cot":     _zeroshot_cot,
    "zeroshot-guided":  _zeroshot_guided,
}


def build_prompt(technique: str, raw_prompt: str) -> str:
    if technique not in TECHNIQUES:
        raise KeyError(f"unknown technique: {technique!r}. Known: {sorted(TECHNIQUES)}")
    return TECHNIQUES[technique](raw_prompt)
