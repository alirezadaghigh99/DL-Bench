# DL-Bench harness

A clean, Docker-based replacement for the legacy evaluator under `src/tester`.

## Why this exists

The original evaluator:
- Hardcoded absolute paths (`/local/data0/moved_data/publishablew/...`) and a
  specific Linux user account.
- Required a hand-built `venv/` per repo, activated by shelling out to
  `bash -c 'source ... && pytest ...'`. Special-cased `conda` for two repos.
- Decided pass/fail by diffing the set of `FAILED` lines in pytest stdout
  before vs. after patching, and maintained a manual ignore-list of flaky
  tests.
- Patched the target function by renaming it to `name1` and inserting a
  wrapper that imported from a sibling `temp.py` written into the repo
  directory. Backups were `.backup` files left next to the source.

This package replaces all of that with:

| concern              | new                                         |
|----------------------|---------------------------------------------|
| environment          | one Docker image per repo (`dlbench/<repo>:latest`) |
| repo registry        | `harness/registry.yaml`                     |
| code extraction      | `harness.extract` — fenced-block aware, AST-validated |
| patching             | `harness.patch` — AST replace, decorators preserved, atomic restore |
| test execution       | `harness.runner` — `docker run` with bind-mounted patched file, junit XML |
| pass/fail decision   | "all listed tests passed" — no diff trick, no flaky-test allow-list |
| orchestration        | `harness.evaluator` + `python -m harness.cli` |

Cross-platform: Windows / macOS / Linux, anywhere Docker runs.

## Layout

```
harness/
├── registry.yaml      # repo → image, repo_dir, defaults
├── registry.py        # loader + RepoSpec dataclass
├── extract.py         # LLM response → Python source
├── patch.py           # AST patch + atomic restore context manager
├── runner.py          # docker subprocess + pytest junit XML parser
├── evaluator.py       # task JSON → TaskResult
├── cli.py             # `python -m harness.cli ...`
├── tests/             # unit tests for the pure-Python pieces
└── examples/
    └── fake_kornia_task.json
```

## Quick start

1. Install Python deps (already in `requirements.txt`):

   ```
   pip install PyYAML pytest
   ```

2. Run the unit tests (no Docker needed):

   ```
   python -m pytest harness/tests -v
   ```

3. Build the image for the repo you want to evaluate. Example for kornia:

   ```
   docker build -t dlbench/kornia:latest dockerfiles/kornia
   ```

4. Run a single task end-to-end:

   ```
   python -m harness.cli single harness/examples/fake_kornia_task.json
   ```

   Output is a single JSON line with `status`, `counts`, `duration_s`, and
   tail of stdout/stderr.

5. Run a folder of LLM-output JSONs (drop-in replacement for the legacy
   `parse_run_combined.py` consumer):

   ```
   python -m harness.cli folder \
     results/llm-output/zeroshot/output_openai-4o_new/v2/kornia \
     results/eval/zeroshot_openai-4o_kornia.jsonl
   ```

## Task JSON shape

The harness consumes the same JSON format that `src/LLM/call.py` already
writes, so existing LLM outputs can be re-evaluated without re-running the
model. Required keys:

- `result` — raw LLM response text.
- `function_name` — symbol to replace.
- `class` — class name, or `""` for a top-level function.
- `ground_truth` — `<package>/<path>/<file>.py` (relative to the repo root in
  the container; trailing `#Lxxx` is stripped).
- `test` — whitespace-separated pytest paths; each may have a
  `::ClassOrSelector` suffix.
- `repo` — must be a key in `registry.yaml`.

Optional metadata (`stage`, `task`, `data`, `prompt`) is preserved in the
result for downstream analysis.

## Result JSON shape

```jsonc
{
  "task_id":     "processed_korniainvert.json",
  "repo":        "kornia",
  "status":      "pass",        // pass | fail | error | timeout
                                // | no_code | no_image | unknown_repo | patch_error
  "detail":      "extracted via fenced-python",
  "counts":      {"passed": 4, "failed": 0, "errored": 0, "skipped": 0},
  "duration_s":  12.4,
  "stdout_tail": "...",
  "stderr_tail": "...",
  "metadata":    {"stage": "...", "task": "...", "data": "..."}
}
```

`status` decision (per task; counts are summed across the listed test paths):

- `pass` — `failed == 0 && errored == 0 && total > 0` for every listed path.
- `fail` — at least one test failed/errored after patching.
- `error` — pytest couldn't even collect tests (often: bad patch import).
- `timeout` — exceeded `defaults.timeout_s` from the registry.
- `no_code` — extractor found no parseable Python in the LLM response.
- `no_image` — `dlbench/<repo>:latest` isn't built locally.
- `unknown_repo` — `repo` is not in `registry.yaml`.
- `patch_error` — function/class wasn't found, or the LLM code didn't define
  the target.

## Adding a new repo

1. Drop a `Dockerfile` in `dockerfiles/<repo>/` that:
   - clones the repo to `/app/<repo>` (the `repo_dir` you'll register),
   - `pip install -e .`s it.
2. Build it: `docker build -t dlbench/<repo>:latest dockerfiles/<repo>`.
3. Add an entry under `repos:` in `harness/registry.yaml`:

   ```yaml
   <repo>:
     image: dlbench/<repo>:latest
     repo_dir: /app/<repo>
   ```

   Override defaults (`pytest_args`, `timeout_s`, `python_bin`) per repo if
   needed.

## What's still TODO

- A lot of the dockerfiles in `dockerfiles/` have not been built yet; only
  `dockerfiles/kornia/Dockerfile` is wired through end-to-end as a worked
  example. The other repos in `registry.yaml` will return `no_image` until
  built.
- `pytorch3d` doesn't have a Dockerfile yet; the legacy code special-cased it
  for conda. Recommended approach: a CUDA-enabled base image (e.g.
  `nvidia/cuda:12.1.0-runtime-ubuntu22.04`) with the same install steps the
  old `src/Builder/pytorch3d/builder.sh` used.
- The legacy modules under `src/tester/` and `src/Builder/` are kept for
  reference but are no longer the entry point. They can be removed once the
  Docker images are built and the historical results are reproduced.
