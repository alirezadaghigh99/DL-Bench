"""End-to-end task evaluation: LLM response in, structured result out.

A task is described by a dict with the same shape that `src/LLM/call.py`
already writes (so this is a drop-in replacement for parse_run_combined.py
on the consumption side):

    {
        "result":         <raw LLM text response>,
        "function_name":  <symbol to replace>,
        "class":          <class name, or "" for top-level functions>,
        "ground_truth":   "<package>/<path>/<file>.py[#Lxxx]",
        "test":           "tests/foo.py[::Selector]"  (whitespace-separated ok),
        "repo":           <key into registry>,
        ...task metadata (stage/task/data) is preserved verbatim
    }

Source files are read out of the container with `docker cp`; the LLM-patched
version is bind-mounted back in for the pytest run. Nothing is written to the
host repo (no .backup files left behind).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .extract import extract
from .patch import PatchError, make_patched_source
from .registry import Registry, RepoSpec, UnknownRepoError
from .runner import RunResult, run_tests


_LINE_SUFFIX = re.compile(r"#L\d+$")


@dataclass
class TaskResult:
    task_id: str
    repo: str
    status: str  # pass | fail | error | timeout | no_code | no_image | unknown_repo | patch_error
    detail: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _strip_line_anchor(p: str) -> str:
    return _LINE_SUFFIX.sub("", p).strip()


def _split_test_spec(t: str) -> tuple[str, str | None]:
    """`tests/foo.py::TestX::test_y` -> ("tests/foo.py", "TestX::test_y")."""
    parts = t.split("::", 1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def _docker_cat(image: str, container_path: str) -> str:
    """Read a file out of an image without persisting a container."""
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image, container_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker cat {container_path} failed: {r.stderr.strip()}")
    return r.stdout


def _tail(s: str, n: int = 4000) -> str:
    return s if len(s) <= n else "...\n" + s[-n:]


def evaluate_task(
    task: dict,
    registry: Registry,
    task_id: str | None = None,
) -> TaskResult:
    repo = task["repo"]
    metadata = {k: task.get(k, "") for k in ("stage", "task", "data")}
    tid = task_id or task.get("filename") or task.get("function_name") or "unnamed"

    try:
        spec: RepoSpec = registry.get(repo)
    except UnknownRepoError as e:
        return TaskResult(tid, repo, "unknown_repo", detail=str(e), metadata=metadata)

    response = task.get("result", "") or ""
    extracted = extract(response)
    if extracted is None:
        return TaskResult(tid, repo, "no_code", detail="no parseable code in response",
                          metadata=metadata)

    src_rel = _strip_line_anchor(task["ground_truth"].split("#")[0])
    container_src = f"{spec.repo_dir.rstrip('/')}/{src_rel.lstrip('/')}"

    try:
        original = _docker_cat(spec.image, container_src)
    except RuntimeError as e:
        return TaskResult(tid, repo, "error", detail=str(e), metadata=metadata)

    function_name = task["function_name"]
    class_name = task.get("class") or None

    try:
        patched = make_patched_source(original, extracted.code, function_name, class_name)
    except PatchError as e:
        return TaskResult(tid, repo, "patch_error", detail=str(e), metadata=metadata)

    test_spec = (task.get("test") or "").split()
    if not test_spec:
        return TaskResult(tid, repo, "error", detail="no test path in task", metadata=metadata)

    # Each task may list >1 test path; treat them as a single pytest invocation.
    test_paths = []
    selectors = []
    for t in test_spec:
        p, sel = _split_test_spec(_strip_line_anchor(t))
        test_paths.append(p)
        if sel:
            selectors.append(f"{p}::{sel}")

    with tempfile.TemporaryDirectory(prefix="dlbench-eval-") as tmp:
        patched_path = Path(tmp) / Path(src_rel).name
        patched_path.write_text(patched, encoding="utf-8")

        # If selectors were given, run them as the targets; otherwise run files.
        # For multi-target invocations we let pytest take all of them.
        result: RunResult | None = None
        targets = selectors if selectors else test_paths
        # Run them one at a time so a single broken test_path doesn't poison the
        # whole result; aggregate.
        agg = RunResult(status="pass")
        for tgt in targets:
            tpath, tsel = _split_test_spec(tgt)
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
            agg.duration_s += r.duration_s
            agg.stdout += r.stdout
            agg.stderr += r.stderr
            # Worst-case status wins.
            order = ["pass", "fail", "error", "timeout", "no_image"]
            if order.index(r.status) > order.index(agg.status):
                agg.status = r.status
        result = agg

    return TaskResult(
        tid, repo,
        status=result.status,
        detail=f"extracted via {extracted.source}",
        counts=asdict(result.counts),
        duration_s=round(result.duration_s, 2),
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
        metadata=metadata,
    )


def evaluate_folder(
    folder: Path,
    registry: Registry,
    out_path: Path,
) -> int:
    """Evaluate every *.json in `folder`, append jsonl results to `out_path`."""
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out:
        for fp in sorted(folder.glob("*.json")):
            with open(fp, encoding="utf-8") as f:
                task = json.load(f)
            res = evaluate_task(task, registry, task_id=fp.name)
            out.write(res.to_jsonl() + "\n")
            out.flush()
            print(f"[{res.status:8s}] {fp.name}")
            n += 1
    return n
