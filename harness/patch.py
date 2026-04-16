"""Patch a target function or class method into a source file.

The old implementation renamed the original function to `name1`, inserted a
wrapper that imported from a sibling `temp.py`, and dropped a backup file next
to the target — leaving the repo dirty if anything crashed mid-run.

This module:
  - Parses the original file with `ast`.
  - Locates the target function (module-level or class method).
  - Replaces its node with the matching definition extracted from the LLM
    response. Preserves any decorators on the original.
  - Injects any new top-level imports from the LLM response that aren't already
    in the file.
  - Returns the new file contents as a string. Writing/restoring is the
    caller's responsibility (see `patched_file` below for a context manager).
"""

from __future__ import annotations

import ast
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class PatchError(Exception):
    pass


def _find_funcdef(tree: ast.Module, function_name: str, class_name: str | None) -> ast.FunctionDef | ast.AsyncFunctionDef:
    if class_name:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name:
                        return item
                raise PatchError(f"function {function_name!r} not found in class {class_name!r}")
        raise PatchError(f"class {class_name!r} not found")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise PatchError(f"top-level function {function_name!r} not found")


def _extract_replacement(
    code: str, function_name: str, class_name: str | None
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[ast.stmt]]:
    """Return (replacement function node, top-level imports from the LLM code)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PatchError(f"LLM code does not parse: {e}") from e

    imports: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]

    # Prefer a class method match if class_name is given and the LLM returned
    # a full class. Otherwise pick the first matching function at any level.
    if class_name:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name:
                        return item, imports

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node, imports

    raise PatchError(f"LLM code did not define {function_name!r}")


def _imports_already_present(tree: ast.Module, new_imports: list[ast.stmt]) -> list[ast.stmt]:
    existing = {ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    return [n for n in new_imports if ast.unparse(n) not in existing]


def make_patched_source(
    original_source: str,
    llm_code: str,
    function_name: str,
    class_name: str | None,
) -> str:
    tree = ast.parse(original_source)
    target = _find_funcdef(tree, function_name, class_name)
    replacement, new_imports = _extract_replacement(llm_code, function_name, class_name)

    # Preserve original decorators (the dataset sometimes includes critical
    # ones like @torch.jit.script that the LLM tends to drop).
    if not replacement.decorator_list:
        replacement.decorator_list = list(target.decorator_list)
    replacement.name = function_name

    def replace_in(body: list[ast.stmt]) -> bool:
        for i, item in enumerate(body):
            if item is target:
                body[i] = replacement
                return True
        return False

    if class_name:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                if replace_in(node.body):
                    break
    else:
        replace_in(tree.body)

    extra = _imports_already_present(tree, new_imports)
    tree.body = extra + tree.body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@contextmanager
def patched_file(path: Path, new_source: str) -> Iterator[Path]:
    """Atomically swap `path` for `new_source`, restoring the original on exit.

    The original is copied to `path.with_suffix(path.suffix + '.dlbench-bak')`
    and the backup is removed on clean exit.
    """
    backup = path.with_suffix(path.suffix + ".dlbench-bak")
    shutil.copy2(path, backup)
    try:
        path.write_text(new_source, encoding="utf-8")
        yield path
    finally:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)
