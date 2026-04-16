from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RepoSpec:
    name: str
    image: str
    repo_dir: str
    python_bin: str
    pytest_args: tuple[str, ...]
    timeout_s: int


class UnknownRepoError(KeyError):
    pass


class Registry:
    def __init__(self, defaults: dict[str, Any], repos: dict[str, dict[str, Any]]):
        self._defaults = defaults
        self._repos = repos

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Registry":
        if path is None:
            path = Path(__file__).parent / "registry.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw.get("defaults", {}) or {}, raw.get("repos", {}) or {})

    def get(self, name: str) -> RepoSpec:
        if name not in self._repos:
            raise UnknownRepoError(
                f"repo {name!r} not registered. Add it to harness/registry.yaml."
            )
        entry = self._repos[name]
        merged = {**self._defaults, **entry}
        return RepoSpec(
            name=name,
            image=merged["image"],
            repo_dir=merged["repo_dir"],
            python_bin=merged.get("python_bin", "python"),
            pytest_args=tuple(merged.get("pytest_args", [])),
            timeout_s=int(merged.get("timeout_s", 300)),
        )

    def names(self) -> list[str]:
        return sorted(self._repos)
