"""Service layer: generate + evaluate, with timing per phase.

Calls the harness primitives directly (extract, patch, run_tests) rather than
the top-level `evaluate_task`, so we can record how long each phase took.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from harness.extract import extract
from harness.patch import PatchError, make_patched_source
from harness.registry import Registry, UnknownRepoError
from harness.runner import RunResult, TestCounts, run_tests

from .llm.base import Generation
from .llm.registry import client_for
from .prompts import build_prompt
from .store import EvaluationRow, GenerationRow, Store, cache_key, now
from .tasks import Task


_LINE_SUFFIX = re.compile(r"#L\d+$")


def _strip_line_anchor(p: str) -> str:
    return _LINE_SUFFIX.sub("", p).strip()


@dataclass
class GenerateResult:
    row: GenerationRow
    from_cache: bool


def generate(
    task: Task, model: str, technique: str, temperature: float, store: Store, *,
    force: bool = False,
) -> GenerateResult:
    """Run (or fetch) a generation. Cached by (model, technique, temperature, prompt)."""
    full_prompt = build_prompt(technique, task.prompt)
    key = cache_key(model, technique, temperature, full_prompt)

    if not force:
        cached = store.get_generation(key)
        if cached is not None:
            return GenerateResult(cached, from_cache=True)

    client = client_for(model)
    gen: Generation = client.generate(model, full_prompt, temperature=temperature)
    row = GenerationRow(
        cache_key=key,
        task_id=task.id,
        model=model,
        technique=technique,
        temperature=temperature,
        prompt=full_prompt,
        response=gen.text,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
        cost_usd=gen.cost_usd,
        duration_s=gen.duration_s,
        created_at=now(),
    )
    store.insert_generation(row)
    return GenerateResult(row, from_cache=False)


def _tail(s: str, n: int = 8000) -> str:
    return s if len(s) <= n else "...\n" + s[-n:]


def _docker_cat(image: str, container_path: str) -> str:
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image, container_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker cat failed: {r.stderr.strip()}")
    return r.stdout


@dataclass
class EvalResult:
    row: EvaluationRow


def evaluate(
    task: Task, generation: GenerationRow, registry: Registry, store: Store,
) -> EvalResult:
    """Run the evaluation, recording per-phase wall time."""
    started = time.monotonic()

    # --- extract ---
    t0 = time.monotonic()
    extracted = extract(generation.response)
    extract_s = time.monotonic() - t0
    if extracted is None:
        return _finish(store, generation.cache_key, "no_code", "no parseable code",
                       TestCounts(), extract_s, 0.0, 0.0, started, "", "")

    # --- resolve repo ---
    try:
        spec = registry.get(task.repo)
    except UnknownRepoError as e:
        return _finish(store, generation.cache_key, "unknown_repo", str(e),
                       TestCounts(), extract_s, 0.0, 0.0, started, "", "")

    src_rel = _strip_line_anchor(task.ground_truth.split("#")[0])
    container_src = f"{spec.repo_dir.rstrip('/')}/{src_rel.lstrip('/')}"

    # --- read original and patch ---
    t0 = time.monotonic()
    try:
        original = _docker_cat(spec.image, container_src)
    except RuntimeError as e:
        patch_s = time.monotonic() - t0
        return _finish(store, generation.cache_key, "no_image", str(e),
                       TestCounts(), extract_s, patch_s, 0.0, started, "", "")

    try:
        patched = make_patched_source(
            original, extracted.code, task.function_name, task.class_name or None,
        )
    except PatchError as e:
        patch_s = time.monotonic() - t0
        return _finish(store, generation.cache_key, "patch_error", str(e),
                       TestCounts(), extract_s, patch_s, 0.0, started, "", "")
    patch_s = time.monotonic() - t0

    # --- docker pytest ---
    t0 = time.monotonic()
    test_specs = (task.test_paths or "").split()
    if not test_specs:
        return _finish(store, generation.cache_key, "error", "no test path",
                       TestCounts(), extract_s, patch_s, 0.0, started, "", "")

    agg = RunResult(status="pass")
    with tempfile.TemporaryDirectory(prefix="dlbench-") as tmp:
        patched_path = Path(tmp) / Path(src_rel).name
        patched_path.write_text(patched, encoding="utf-8")

        for raw in test_specs:
            clean = _strip_line_anchor(raw)
            parts = clean.split("::", 1)
            tpath = parts[0]
            tsel = parts[1] if len(parts) > 1 else None
            r = run_tests(
                spec=spec,
                patched_src_host=patched_path,
                src_path_in_repo=src_rel,
                test_path_in_repo=tpath,
                test_selector=tsel,
            )
            agg.counts.passed  += r.counts.passed
            agg.counts.failed  += r.counts.failed
            agg.counts.errored += r.counts.errored
            agg.counts.skipped += r.counts.skipped
            agg.stdout += r.stdout
            agg.stderr += r.stderr
            order = ["pass", "fail", "error", "timeout", "no_image"]
            if order.index(r.status) > order.index(agg.status):
                agg.status = r.status
    docker_s = time.monotonic() - t0

    return _finish(store, generation.cache_key, agg.status,
                   f"extracted via {extracted.source}",
                   agg.counts, extract_s, patch_s, docker_s, started,
                   _tail(agg.stdout), _tail(agg.stderr))


def _finish(
    store: Store, key: str, status: str, detail: str, counts: TestCounts,
    extract_s: float, patch_s: float, docker_s: float, started: float,
    stdout_tail: str, stderr_tail: str,
) -> EvalResult:
    row = EvaluationRow(
        id=0, cache_key=key, status=status, detail=detail,
        counts={"passed": counts.passed, "failed": counts.failed,
                "errored": counts.errored, "skipped": counts.skipped},
        extract_s=round(extract_s, 3),
        patch_s=round(patch_s, 3),
        docker_s=round(docker_s, 3),
        total_s=round(time.monotonic() - started, 3),
        stdout_tail=stdout_tail, stderr_tail=stderr_tail,
        created_at=now(),
    )
    row.id = store.insert_evaluation(row)
    return EvalResult(row)
