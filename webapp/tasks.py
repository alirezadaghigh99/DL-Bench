"""Task index over data/DL-Bench-Enriched-Processed-Sorted.csv.

Loaded once into memory at startup (the CSV is ~19k rows, well under 1 GB
even with the test_suites field). We index by deterministic id =
sha1(repo + filename) so URLs are stable across restarts.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


@dataclass
class Task:
    id: str
    repo: str
    filename: str
    function_name: str
    class_name: str
    ground_truth: str       # path within repo
    test_paths: str         # whitespace-separated
    prompt: str
    stage: str
    task_type: str
    data_type: str

    def to_harness_payload(self, llm_response: str) -> dict:
        """Shape that harness.evaluate_task expects."""
        return {
            "result":        llm_response,
            "function_name": self.function_name,
            "class":         self.class_name,
            "ground_truth":  self.ground_truth,
            "test":          self.test_paths,
            "repo":          self.repo,
            "stage":         self.stage,
            "task":          self.task_type,
            "data":          self.data_type,
            "filename":      self.filename,
        }


def task_id(repo: str, filename: str) -> str:
    return hashlib.sha1(f"{repo}|{filename}".encode()).hexdigest()[:16]


class TaskIndex:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._tasks: dict[str, Task] = {}
        self._by_repo: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                repo = row["repo"]
                filename = row["filename"]
                tid = task_id(repo, filename)
                t = Task(
                    id=tid,
                    repo=repo,
                    filename=filename,
                    function_name=row.get("function", "") or "",
                    class_name=row.get("class", "") or "",
                    ground_truth=row.get("ground Truth", "") or "",
                    test_paths=row.get("test_cases", "") or "",
                    prompt=row.get("prompt", "") or "",
                    stage=row.get("stage", "") or "",
                    task_type=row.get("task", "") or "",
                    data_type=row.get("data", "") or "",
                )
                self._tasks[tid] = t
                self._by_repo.setdefault(repo, []).append(tid)

    def get(self, tid: str) -> Task | None:
        return self._tasks.get(tid)

    def repos(self) -> list[str]:
        return sorted(self._by_repo)

    def list(self, repo: str | None = None, page: int = 0, page_size: int = 50) -> tuple[list[Task], int]:
        if repo:
            ids = self._by_repo.get(repo, [])
        else:
            ids = list(self._tasks)
        total = len(ids)
        start = page * page_size
        end = start + page_size
        page_ids = ids[start:end]
        return [self._tasks[i] for i in page_ids], total

    def __len__(self) -> int:
        return len(self._tasks)
