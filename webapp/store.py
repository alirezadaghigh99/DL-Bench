"""SQLite-backed store for generations and evaluations.

Cache key for a generation = sha256(model + technique + temperature + prompt).
Re-running the same (task, model, technique) is therefore free.

Schema is created lazily on first connect.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_LOCK = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    cache_key      TEXT PRIMARY KEY,
    task_id        TEXT NOT NULL,
    model          TEXT NOT NULL,
    technique      TEXT NOT NULL,
    temperature    REAL NOT NULL,
    prompt         TEXT NOT NULL,
    response       TEXT NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    cost_usd       REAL NOT NULL,
    duration_s     REAL NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gen_task ON generations(task_id);

CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key       TEXT NOT NULL,
    status          TEXT NOT NULL,
    detail          TEXT,
    counts_json     TEXT NOT NULL,
    extract_s       REAL NOT NULL,
    patch_s         REAL NOT NULL,
    docker_s        REAL NOT NULL,
    total_s         REAL NOT NULL,
    stdout_tail     TEXT,
    stderr_tail     TEXT,
    created_at      REAL NOT NULL,
    FOREIGN KEY (cache_key) REFERENCES generations(cache_key)
);
CREATE INDEX IF NOT EXISTS idx_eval_key ON evaluations(cache_key);
"""


@dataclass
class GenerationRow:
    cache_key: str
    task_id: str
    model: str
    technique: str
    temperature: float
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float
    created_at: float


@dataclass
class EvaluationRow:
    id: int
    cache_key: str
    status: str
    detail: str
    counts: dict[str, int]
    extract_s: float
    patch_s: float
    docker_s: float
    total_s: float
    stdout_tail: str
    stderr_tail: str
    created_at: float


def cache_key(model: str, technique: str, temperature: float, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(technique.encode())
    h.update(b"\0")
    h.update(f"{temperature:.4f}".encode())
    h.update(b"\0")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with _LOCK:
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            try:
                yield con
                con.commit()
            finally:
                con.close()

    # ----- generations -----

    def get_generation(self, key: str) -> GenerationRow | None:
        with self._connect() as c:
            r = c.execute("SELECT * FROM generations WHERE cache_key = ?", (key,)).fetchone()
            return _row_to_gen(r) if r else None

    def get_generation_for_task(self, task_id: str, model: str, technique: str) -> GenerationRow | None:
        with self._connect() as c:
            r = c.execute(
                "SELECT * FROM generations WHERE task_id = ? AND model = ? AND technique = ? ORDER BY created_at DESC LIMIT 1",
                (task_id, model, technique),
            ).fetchone()
            return _row_to_gen(r) if r else None

    def list_generations_for_task(self, task_id: str) -> list[GenerationRow]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM generations WHERE task_id = ? ORDER BY created_at DESC",
                (task_id,),
            ).fetchall()
            return [_row_to_gen(r) for r in rows]

    def insert_generation(self, row: GenerationRow) -> None:
        with self._connect() as c:
            c.execute(
                """INSERT OR REPLACE INTO generations
                   (cache_key, task_id, model, technique, temperature, prompt, response,
                    input_tokens, output_tokens, cost_usd, duration_s, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row.cache_key, row.task_id, row.model, row.technique, row.temperature,
                 row.prompt, row.response, row.input_tokens, row.output_tokens,
                 row.cost_usd, row.duration_s, row.created_at),
            )

    # ----- evaluations -----

    def insert_evaluation(self, row: EvaluationRow) -> int:
        with self._connect() as c:
            cur = c.execute(
                """INSERT INTO evaluations
                   (cache_key, status, detail, counts_json, extract_s, patch_s, docker_s,
                    total_s, stdout_tail, stderr_tail, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (row.cache_key, row.status, row.detail, json.dumps(row.counts),
                 row.extract_s, row.patch_s, row.docker_s, row.total_s,
                 row.stdout_tail, row.stderr_tail, row.created_at),
            )
            return cur.lastrowid

    def list_evaluations(self, cache_key: str) -> list[EvaluationRow]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM evaluations WHERE cache_key = ? ORDER BY created_at DESC",
                (cache_key,),
            ).fetchall()
            return [_row_to_eval(r) for r in rows]

    # ----- aggregate -----

    def totals(self) -> dict[str, Any]:
        with self._connect() as c:
            g = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(input_tokens),0) it, COALESCE(SUM(output_tokens),0) ot, COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(duration_s),0) dur FROM generations"
            ).fetchone()
            e = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(total_s),0) dur, COALESCE(SUM(CASE WHEN status='pass' THEN 1 ELSE 0 END), 0) p FROM evaluations"
            ).fetchone()
            return {
                "generations": g["n"],
                "input_tokens": g["it"],
                "output_tokens": g["ot"],
                "cost_usd": g["cost"],
                "gen_duration_s": g["dur"],
                "evaluations": e["n"],
                "eval_duration_s": e["dur"],
                "eval_passed": e["p"],
            }


def _row_to_gen(r: sqlite3.Row) -> GenerationRow:
    return GenerationRow(
        cache_key=r["cache_key"], task_id=r["task_id"], model=r["model"],
        technique=r["technique"], temperature=r["temperature"], prompt=r["prompt"],
        response=r["response"], input_tokens=r["input_tokens"],
        output_tokens=r["output_tokens"], cost_usd=r["cost_usd"],
        duration_s=r["duration_s"], created_at=r["created_at"],
    )


def _row_to_eval(r: sqlite3.Row) -> EvaluationRow:
    return EvaluationRow(
        id=r["id"], cache_key=r["cache_key"], status=r["status"], detail=r["detail"] or "",
        counts=json.loads(r["counts_json"]), extract_s=r["extract_s"], patch_s=r["patch_s"],
        docker_s=r["docker_s"], total_s=r["total_s"],
        stdout_tail=r["stdout_tail"] or "", stderr_tail=r["stderr_tail"] or "",
        created_at=r["created_at"],
    )


def now() -> float:
    return time.time()
