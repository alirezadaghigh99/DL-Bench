"""DL-Bench webapp: pick a task, generate, evaluate, see costs/timings/bugs."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

from harness.registry import Registry

from . import services
from .llm.registry import known_models
from .prompts import TECHNIQUES
from .store import Store
from .tasks import TaskIndex


def create_app(
    csv_path: str | Path | None = None,
    db_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    csv_path = Path(csv_path or os.environ.get(
        "DLBENCH_CSV", "data/DL-Bench-Enriched-Processed-Sorted.csv"))
    db_path = Path(db_path or os.environ.get("DLBENCH_DB", "var/dlbench.sqlite"))
    registry_path = registry_path or os.environ.get("DLBENCH_REGISTRY")

    app.config["TASKS"] = TaskIndex(csv_path)
    app.config["STORE"] = Store(db_path)
    app.config["REGISTRY"] = Registry.load(registry_path)

    @app.template_filter("usd")
    def usd(v):
        return f"${v:.6f}" if v < 0.01 else f"${v:.4f}"

    @app.template_filter("dur")
    def dur(v):
        if v is None:
            return "—"
        if v < 1:
            return f"{v*1000:.0f} ms"
        if v < 60:
            return f"{v:.2f} s"
        return f"{int(v // 60)}m {int(v % 60)}s"

    @app.route("/")
    def index():
        idx: TaskIndex = app.config["TASKS"]
        store: Store = app.config["STORE"]
        return render_template(
            "index.html",
            n_tasks=len(idx),
            repos=idx.repos(),
            totals=store.totals(),
        )

    @app.route("/tasks")
    def tasks():
        idx: TaskIndex = app.config["TASKS"]
        repo = request.args.get("repo") or None
        page = max(int(request.args.get("page", 0)), 0)
        page_size = 50
        items, total = idx.list(repo=repo, page=page, page_size=page_size)
        return render_template(
            "tasks.html",
            items=items, total=total, page=page, page_size=page_size,
            repo=repo, repos=idx.repos(),
        )

    @app.route("/task/<tid>", methods=["GET", "POST"])
    def task_view(tid: str):
        idx: TaskIndex = app.config["TASKS"]
        store: Store = app.config["STORE"]
        registry: Registry = app.config["REGISTRY"]
        task = idx.get(tid)
        if task is None:
            abort(404)

        message = None
        if request.method == "POST":
            action = request.form.get("action")
            model = request.form.get("model", "gpt-4o")
            technique = request.form.get("technique", "zeroshot")
            temperature = float(request.form.get("temperature", "0"))
            force = bool(request.form.get("force"))
            try:
                if action == "generate":
                    services.generate(task, model, technique, temperature, store, force=force)
                    return redirect(url_for("task_view", tid=tid,
                                            model=model, technique=technique))
                elif action == "evaluate":
                    g = store.get_generation_for_task(task.id, model, technique)
                    if g is None:
                        message = "No generation yet for this (model, technique). Generate first."
                    else:
                        services.evaluate(task, g, registry, store)
                        return redirect(url_for("task_view", tid=tid,
                                                model=model, technique=technique))
            except Exception as e:
                message = f"{type(e).__name__}: {e}"

        model = request.args.get("model", "gpt-4o")
        technique = request.args.get("technique", "zeroshot")
        gens = store.list_generations_for_task(task.id)
        active_gen = store.get_generation_for_task(task.id, model, technique)
        evals = store.list_evaluations(active_gen.cache_key) if active_gen else []

        return render_template(
            "task.html",
            task=task,
            models=known_models(),
            techniques=sorted(TECHNIQUES),
            model=model,
            technique=technique,
            generations=gens,
            active_gen=active_gen,
            evaluations=evals,
            registered_repos=set(registry.names()),
            message=message,
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000, debug=True)
