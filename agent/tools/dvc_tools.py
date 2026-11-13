"""DVC tools for the agent. Only whitelisted, read-only commands are executed."""
from __future__ import annotations

import functools
import logging
import re
import subprocess
import time
from pathlib import Path

import yaml

log = logging.getLogger("agent.tools.dvc")
REPO = Path(__file__).resolve().parents[2]

# Approved command prefixes. Nothing that changes data or git state is allowed.
ALLOWED = {
    ("dvc", "status"),
    ("dvc", "data", "status"),
    ("git", "rev-parse"),
    ("git", "tag"),
    ("git", "show"),
    ("git", "log"),
}
SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    if tuple(args[:2]) not in ALLOWED and tuple(args[:3]) not in ALLOWED:
        raise PermissionError(f"Command not approved: {' '.join(args[:3])}")
    start = time.time()
    proc = subprocess.run(
        args, check=False, cwd=REPO, capture_output=True, text=True, timeout=timeout, shell=False
    )
    log.info(
        "cmd=%s rc=%s duration_ms=%d",
        " ".join(args[:3]), proc.returncode, (time.time() - start) * 1000,
    )
    return proc


def _check_ref(ref: str) -> str:
    if not SAFE_REF.match(ref):
        raise ValueError(f"Invalid git ref: {ref!r}")
    return ref


def _check_path(path: str) -> str:
    if ".." in path or path.startswith("/") or not path.endswith(".dvc"):
        raise ValueError(f"Path must be a relative .dvc file: {path!r}")
    return path


def tool(fn):
    """Return structured {status, data|error} and never raise."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return {"status": "ok", "data": fn(*args, **kwargs)}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"{fn.__name__} timed out"}
        except PermissionError as e:
            return {"status": "denied", "error": str(e)}
        except (ValueError, FileNotFoundError, KeyError) as e:
            return {"status": "invalid", "error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    return wrapper


def _dvc_out_at(ref: str, path: str) -> dict:
    """Read the .dvc file as it was at a git ref."""
    proc = _run(["git", "show", f"{_check_ref(ref)}:{_check_path(path)}"])
    if proc.returncode != 0:
        raise ValueError(f"Cannot read {path} at {ref}: {proc.stderr.strip()}")
    return yaml.safe_load(proc.stdout)["outs"][0]


def _tags_for_md5(md5: str, path: str) -> list[str]:
    tags = _run(["git", "tag", "--list"]).stdout.split()
    found = []
    for t in tags:
        try:
            if _dvc_out_at(t, path)["md5"] == md5:
                found.append(t)
        except ValueError:
            continue
    return found


@tool
def list_datasets() -> list[dict]:
    """List all DVC-tracked datasets in the repo."""
    out = []
    for f in sorted(REPO.rglob("*.dvc")):
        if ".venv" in f.parts or ".dvc" in f.parts:
            continue
        o = yaml.safe_load(f.read_text())["outs"][0]
        out.append({
            "dvc_file": str(f.relative_to(REPO)),
            "data_path": o["path"], "md5": o["md5"], "size_bytes": o["size"],
        })
    return out


@tool
def get_dataset_version(path: str = "data/dataset.csv.dvc") -> dict:
    """Current md5, matching git tag(s) and the last git commit touching the .dvc file."""
    o = yaml.safe_load((REPO / _check_path(path)).read_text())["outs"][0]
    commit = _run(["git", "log", "-1", "--format=%H", "--", path]).stdout.strip()
    return {
        "dvc_file": path, "md5": o["md5"], "size_bytes": o["size"],
        "git_commit": commit, "git_tags": _tags_for_md5(o["md5"], path),
    }


@tool
def compare_versions(ref_a: str, ref_b: str, path: str = "data/dataset.csv.dvc") -> dict:
    """Compare the dataset at two git refs (tags or commits)."""
    a, b = _dvc_out_at(ref_a, path), _dvc_out_at(ref_b, path)
    return {
        ref_a: {"md5": a["md5"], "size_bytes": a["size"]},
        ref_b: {"md5": b["md5"], "size_bytes": b["size"]},
        "changed": a["md5"] != b["md5"],
        "size_diff_bytes": b["size"] - a["size"],
    }


@tool
def detect_changes() -> dict:
    """Has tracked data changed compared to what git/DVC recorded?"""
    s = _run(["dvc", "status", "--json"])
    d = _run(["dvc", "data", "status", "--json"])
    return {
        "dvc_status": s.stdout.strip() or "{}",
        "dvc_data_status": d.stdout.strip() or "{}",
        "up_to_date": s.stdout.strip() in ("", "{}"),
    }


@tool
def explain_reproduce(ref: str, path: str = "data/dataset.csv.dvc") -> dict:
    """Explain (not execute) how to reproduce the dataset of a past version."""
    _dvc_out_at(ref, path)  # validates that ref and file exist
    return {
        "steps": [
            f"git checkout {ref} -- {path}",
            f"dvc checkout {path}",
            "If the file is missing locally: dvc pull",
            f"Return to latest: git checkout main -- {path} && dvc checkout {path}",
        ],
        "note": "Commands are shown only. The agent does not modify the workspace.",
    }


@tool
def get_experiment_dataset(run_id: str, tracking_uri: str = "http://127.0.0.1:5000") -> dict:
    """Read recorded dataset provenance from an MLflow run's tags."""
    from mlflow import MlflowClient

    tags = MlflowClient(tracking_uri=tracking_uri).get_run(run_id).data.tags
    md5 = tags.get("dvc_md5")
    if not md5:
        raise KeyError(f"Run {run_id} has no recorded dvc_md5 tag")
    return {
        "run_id": run_id,
        "dataset_version": tags.get("dataset_version"),
        "dvc_md5": md5,
        "git_commit": tags.get("git_commit"),
        "matching_git_tags": _tags_for_md5(md5, "data/dataset.csv.dvc"),
    }