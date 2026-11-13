"""MLflow tools for the agent. Uses MlflowClient only (no UI scraping)."""
from __future__ import annotations

import os

# Fail fast when the server is down instead of retrying silently for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

from mlflow import MlflowClient

from agent.tools.common import tool

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=TRACKING_URI)


def _experiment_id(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise KeyError(f"Experiment not found: {name!r}")
    return exp.experiment_id


def _summary(run) -> dict:
    tags = run.data.tags
    return {
        "run_id": run.info.run_id,
        "name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "dataset_version": tags.get("dataset_version"),
        "dvc_md5": tags.get("dvc_md5"),
        "git_commit": tags.get("git_commit"),
    }


def _recent_finished(client, experiment_name: str, last_n: int) -> list:
    exp_id = _experiment_id(client, experiment_name)
    return client.search_runs(
        [exp_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=last_n,
    )


@tool
def list_experiments() -> list[dict]:
    """List all MLflow experiments."""
    return [
        {"experiment_id": e.experiment_id, "name": e.name}
        for e in _client().search_experiments()
    ]


@tool
def list_runs(experiment_name: str, limit: int = 10) -> list[dict]:
    """List the most recent runs (any status) of an experiment."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    client = _client()
    runs = client.search_runs(
        [_experiment_id(client, experiment_name)],
        order_by=["attributes.start_time DESC"],
        max_results=limit,
    )
    return [_summary(r) for r in runs]


@tool
def get_run_details(run_id: str) -> dict:
    """Params, metrics, provenance tags and artifact list of one run."""
    client = _client()
    run = client.get_run(run_id)
    out = _summary(run)
    out["artifacts"] = [a.path for a in client.list_artifacts(run_id)]
    return out


@tool
def find_best_run(experiment_name: str, metric: str = "f1", limit: int = 50) -> dict:
    """Best FINISHED run by a metric (highest wins)."""
    client = _client()
    runs = client.search_runs(
        [_experiment_id(client, experiment_name)],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=limit,
    )
    runs = [r for r in runs if metric in r.data.metrics]
    if not runs:
        raise KeyError(f"No finished runs with metric {metric!r}")
    return {"metric": metric, "best": _summary(runs[0]), "runs_considered": len(runs)}


@tool
def compare_runs(run_ids: list[str]) -> dict:
    """Side-by-side comparison of two or more runs."""
    if len(run_ids) < 2:
        raise ValueError("Provide at least two run ids")
    client = _client()
    return {"runs": [_summary(client.get_run(r)) for r in run_ids]}


@tool
def get_model_versions(model_name: str | None = None) -> dict:
    """Registry info: registered models and their versions (may be empty)."""
    client = _client()
    models = client.search_registered_models()
    if model_name:
        models = [m for m in models if m.name == model_name]
    out = []
    for m in models:
        versions = client.search_model_versions(f"name='{m.name}'")
        out.append({
            "name": m.name,
            "versions": [
                {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                for v in versions
            ],
        })
    return {"registered_models": out, "registry_empty": not out}


@tool
def recommend_candidate(experiment_name: str, metric: str = "f1", last_n: int = 5) -> dict:
    """Compare the last N finished runs and pick a candidate WITH caveats.

    The highest metric is a candidate, not proof of production readiness.
    """
    client = _client()
    runs = [_summary(r) for r in _recent_finished(client, experiment_name, last_n)]
    runs = [r for r in runs if metric in r["metrics"]]
    if not runs:
        raise KeyError(f"No finished runs with metric {metric!r}")
    ranked = sorted(runs, key=lambda r: r["metrics"][metric], reverse=True)
    best = ranked[0]

    caveats = []
    if not best["dvc_md5"]:
        caveats.append("Best run has no recorded dataset provenance.")
    versions = {r["dataset_version"] for r in runs}
    if len(versions) > 1:
        caveats.append(
            f"Runs use different dataset versions {sorted(v or 'unknown' for v in versions)}; "
            "metrics are not directly comparable."
        )
    if len(ranked) > 1:
        gap = best["metrics"][metric] - ranked[1]["metrics"][metric]
        if gap < 0.01:
            caveats.append(f"Margin over runner-up is only {gap:.4f}; difference may be noise.")
    if len(runs) < last_n:
        caveats.append(f"Only {len(runs)} finished runs available (asked for {last_n}).")
    registry = get_model_versions()
    if registry["status"] == "ok" and registry["data"]["registry_empty"]:
        caveats.append("No model is registered; nothing has passed a promotion step.")
    caveats.append("Only offline metrics on one test split; no deployment or drift evidence.")

    return {
        "metric": metric,
        "candidate": best,
        "ranking": [
            {"run_id": r["run_id"], "name": r["name"], metric: r["metrics"][metric],
             "dataset_version": r["dataset_version"]}
            for r in ranked
        ],
        "caveats": caveats,
    }