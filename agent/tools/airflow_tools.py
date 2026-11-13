"""Airflow 3 tools for the agent. REST API with JWT auth. Only approved DAGs can be triggered."""
from __future__ import annotations

import os
import time

import requests

from agent.tools.common import tool

BASE_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
APPROVED_DAGS = {"training_pipeline"}
TIMEOUT = 10

_token: dict = {"value": None, "exp": 0.0}


def _get_token() -> str:
    if _token["value"] and time.time() < _token["exp"]:
        return _token["value"]
    r = requests.post(
        f"{BASE_URL}/auth/token",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _token.update(value=r.json()["access_token"], exp=time.time() + 600)
    return _token["value"]


def _api(method: str, path: str, **kwargs):
    headers = {"Authorization": f"Bearer {_get_token()}"}
    r = requests.request(
        method, f"{BASE_URL}/api/v2{path}", headers=headers, timeout=TIMEOUT, **kwargs
    )
    if r.status_code == 404:
        raise KeyError(f"Not found: {path}")
    r.raise_for_status()
    return r.json()


def _check_dag(dag_id: str) -> str:
    if dag_id not in APPROVED_DAGS:
        raise PermissionError(f"DAG {dag_id!r} is not approved. Allowed: {sorted(APPROVED_DAGS)}")
    return dag_id


@tool
def list_dags() -> list[dict]:
    """List all DAGs."""
    data = _api("GET", "/dags")
    return [
        {"dag_id": d["dag_id"], "is_paused": d["is_paused"], "approved": d["dag_id"] in APPROVED_DAGS}
        for d in data["dags"]
    ]


@tool
def get_dag_runs(dag_id: str = "training_pipeline", limit: int = 5) -> list[dict]:
    """Recent runs of a DAG, newest first."""
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    data = _api("GET", f"/dags/{dag_id}/dagRuns", params={"limit": limit, "order_by": "-start_date"})
    return [
        {
            "run_id": r["dag_run_id"],
            "state": r["state"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "conf": r.get("conf"),
        }
        for r in data["dag_runs"]
    ]


@tool
def get_task_instances(dag_id: str, run_id: str) -> list[dict]:
    """Task states of one run."""
    data = _api("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
    return [
        {
            "task_id": t["task_id"],
            "state": t["state"],
            "try_number": t["try_number"],
            "start_date": t["start_date"],
            "end_date": t["end_date"],
        }
        for t in data["task_instances"]
    ]


@tool
def find_failed_tasks(dag_id: str = "training_pipeline", run_id: str | None = None) -> dict:
    """Failed tasks (and upstream_failed) of a run; defaults to the latest failed run."""
    if run_id is None:
        runs = get_dag_runs(dag_id, 10)
        if runs["status"] != "ok":
            raise RuntimeError(runs["error"])
        failed = [r for r in runs["data"] if r["state"] == "failed"]
        if not failed:
            return {"run_id": None, "failed_tasks": [], "note": "No failed runs in the last 10."}
        run_id = failed[0]["run_id"]
    tis = get_task_instances(dag_id, run_id)
    if tis["status"] != "ok":
        raise RuntimeError(tis["error"])
    return {
        "run_id": run_id,
        "failed_tasks": [t for t in tis["data"] if t["state"] in ("failed", "upstream_failed")],
    }


@tool
def get_task_diagnostics(dag_id: str, run_id: str, task_id: str, try_number: int = 1) -> dict:
    """Error message and last log lines of a task attempt."""
    data = _api(
        "GET",
        f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}",
        params={"full_content": "true"},
    )
    lines = data.get("content", [])
    text = [c if isinstance(c, str) else c.get("event", "") for c in lines]
    errors = []
    for c in lines:
        if isinstance(c, dict) and c.get("level") == "error":
            errors.append(c.get("event", ""))
            for d in c.get("error_detail", []) or []:
                errors.append(f"{d.get('exc_type')}: {d.get('exc_value')}")
    return {"task_id": task_id, "try_number": try_number, "errors": errors[-5:], "tail": text[-5:]}


@tool
def trigger_training(conf: dict | None = None, confirmed: bool = False) -> dict:
    """Trigger the approved training DAG. Requires explicit confirmation."""
    dag_id = _check_dag("training_pipeline")
    if not confirmed:
        raise PermissionError("Triggering a DAG needs explicit user confirmation (confirmed=True).")
    conf = conf or {}
    allowed = {"n_estimators", "max_depth", "dataset_version", "min_f1"}
    bad = set(conf) - allowed
    if bad:
        raise ValueError(f"Unsupported conf keys: {sorted(bad)}")
    r = _api("POST", f"/dags/{dag_id}/dagRuns", json={"logical_date": None, "conf": conf})
    return {"run_id": r["dag_run_id"], "state": r["state"]}