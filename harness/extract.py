"""Extract Python code from an LLM response.

Strategy, in order:
  1. The longest fenced ``` python ``` block.
  2. The longest plain ``` ``` block whose contents parse as Python.
  3. The whole response if it parses as Python.
  4. None.

We deliberately do not try to "salvage" half-parseable text. If the model
didn't return code we can run, callers should record that as `no_code` rather
than feed the test harness garbage.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_FENCED = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Extracted:
    code: str
    source: str  # "fenced-python" | "fenced-plain" | "raw"


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def extract(response: str) -> Extracted | None:
    if not response:
        return None

    blocks = _FENCED.findall(response)
    py_blocks = [c for tag, c in blocks if tag.lower() in {"python", "py"}]
    plain_blocks = [c for tag, c in blocks if tag == ""]

    py_blocks.sort(key=len, reverse=True)
    for c in py_blocks:
        if _parses(c):
            return Extracted(c, "fenced-python")

    plain_blocks.sort(key=len, reverse=True)
    for c in plain_blocks:
        if _parses(c):
            return Extracted(c, "fenced-plain")

    if _parses(response):
        return Extracted(response, "raw")

    return None
