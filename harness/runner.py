"""Run pytest inside the per-repo Docker image and return a structured result.

Replaces the bash + venv-source mess in src/tester/test_runner_*.py.

Volume strategy:
  - The patched source file is bind-mounted read-only over its location inside
    the image (so the editable install picks it up automatically).
  - If the dataset bundles the test file (test_suites field), we mount that
    too; otherwise we just run the test path that already exists in the image.
  - Pytest writes a junit XML to /tmp/dlbench/result.xml inside the container,
    which we mount to a host temp dir and parse.

The runner intentionally has no knowledge of the dataset or LLM. It takes
already-prepared files and returns a Result.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .registry import RepoSpec


@dataclass
class TestCounts:
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errored + self.skipped


@dataclass
class RunResult:
    status: str  # "pass" | "fail" | "error" | "timeout" | "no_image"
    counts: TestCounts = field(default_factory=TestCounts)
    duration_s: float = 0.0
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_exists(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _parse_junit(path: Path) -> TestCounts:
    counts = TestCounts()
    if not path.exists():
        return counts
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return counts
    # JUnit XML: <testsuites><testsuite tests=.. failures=.. errors=.. skipped=..>
    suites = root.findall(".//testsuite") or [root]
    for s in suites:
        tests = int(s.get("tests", 0))
        failures = int(s.get("failures", 0))
        errors = int(s.get("errors", 0))
        skipped = int(s.get("skipped", 0))
        counts.failed += failures
        counts.errored += errors
        counts.skipped += skipped
        counts.passed += max(tests - failures - errors - skipped, 0)
    return counts


def run_tests(
    spec: RepoSpec,
    patched_src_host: Path,
    src_path_in_repo: str,
    test_path_in_repo: str,
    test_file_host: Path | None = None,
    test_selector: str | None = None,
) -> RunResult:
    """Run pytest in the per-repo container.

    Args:
        spec: registry entry for the repo.
        patched_src_host: file on the host containing the LLM-patched source.
        src_path_in_repo: path of the source file relative to spec.repo_dir
            (e.g. "kornia/enhance/adjust.py").
        test_path_in_repo: path of the test file relative to spec.repo_dir.
        test_file_host: optional override for the test file (mounted in).
        test_selector: optional ::TestClass::test_method suffix.
    """
    if not _docker_available():
        return RunResult(status="error", stderr="docker not on PATH")
    if not _image_exists(spec.image):
        return RunResult(status="no_image", stderr=f"image {spec.image} not built")

    container_src = f"{spec.repo_dir.rstrip('/')}/{src_path_in_repo.lstrip('/')}"
    container_test = f"{spec.repo_dir.rstrip('/')}/{test_path_in_repo.lstrip('/')}"
    pytest_target = container_test + (f"::{test_selector}" if test_selector else "")

    with tempfile.TemporaryDirectory(prefix="dlbench-") as host_tmp:
        host_results = Path(host_tmp) / "results"
        host_results.mkdir()

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{patched_src_host.resolve()}:{container_src}:ro",
            "-v", f"{host_results.resolve()}:/dlbench-results:rw",
        ]
        if test_file_host is not None:
            cmd += ["-v", f"{test_file_host.resolve()}:{container_test}:ro"]
        cmd += [
            "-w", spec.repo_dir,
            spec.image,
            spec.python_bin, "-m", "pytest",
            pytest_target,
            "--junitxml=/dlbench-results/junit.xml",
            *spec.pytest_args,
        ]

        start = time.monotonic()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=spec.timeout_s)
        except subprocess.TimeoutExpired as e:
            return RunResult(
                status="timeout",
                duration_s=time.monotonic() - start,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            )
        duration = time.monotonic() - start

        counts = _parse_junit(host_results / "junit.xml")
        if r.returncode == 0 and counts.failed == 0 and counts.errored == 0 and counts.total > 0:
            status = "pass"
        elif counts.failed > 0 or counts.errored > 0:
            status = "fail"
        elif counts.total == 0:
            status = "error"  # tests didn't even collect
        else:
            status = "fail"

        return RunResult(
            status=status,
            counts=counts,
            duration_s=duration,
            stdout=r.stdout,
            stderr=r.stderr,
            exit_code=r.returncode,
        )
