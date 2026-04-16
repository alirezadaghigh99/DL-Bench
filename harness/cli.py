"""Command-line entry. Run a single task or a whole folder.

    python -m harness.cli single path/to/task.json
    python -m harness.cli folder path/to/llm-output-dir results.jsonl
    python -m harness.cli list-repos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluator import evaluate_folder, evaluate_task
from .registry import Registry


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("dlbench")
    p.add_argument("--registry", type=Path, default=None,
                   help="Override path to registry.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single", help="Evaluate one task JSON.")
    single.add_argument("task_json", type=Path)

    folder = sub.add_parser("folder", help="Evaluate every *.json in a folder.")
    folder.add_argument("folder", type=Path)
    folder.add_argument("out_jsonl", type=Path)

    sub.add_parser("list-repos", help="Print registered repo names.")

    args = p.parse_args(argv)
    registry = Registry.load(args.registry)

    if args.cmd == "list-repos":
        for n in registry.names():
            print(n)
        return 0

    if args.cmd == "single":
        task = json.loads(args.task_json.read_text(encoding="utf-8"))
        res = evaluate_task(task, registry, task_id=args.task_json.name)
        print(res.to_jsonl())
        return 0 if res.status == "pass" else 1

    if args.cmd == "folder":
        n = evaluate_folder(args.folder, registry, args.out_jsonl)
        print(f"evaluated {n} tasks; wrote {args.out_jsonl}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
