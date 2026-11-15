import subprocess

import pytest
import requests

from agent.tools import airflow_tools as af
from agent.tools import dvc_tools as dv
from agent.tools import kubernetes_tools as k8
from agent.tools import mlflow_tools as ml
from agent.tools.common import tool


def test_tool_wrapper_maps_timeout_to_status():
    @tool
    def slow():
        raise subprocess.TimeoutExpired("cmd", 1)

    assert slow()["status"] == "timeout"


def test_mlflow_server_down_returns_error_status(monkeypatch):
    def boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(ml, "_client", boom)
    out = ml.find_best_run("mlops-copilot-v3")
    assert out["status"] == "error" and "connection refused" in out["error"]


def test_airflow_timeout_is_reported(monkeypatch):
    def timeout(*a, **kw):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(af, "_get_token", lambda: "t")
    monkeypatch.setattr(af.requests, "request", timeout)
    out = af.list_dags()
    assert out["status"] == "error" and "timed out" in out["error"]


def test_airflow_trigger_needs_confirmation_and_valid_conf():
    assert af.trigger_training()["status"] == "denied"
    assert af.trigger_training(confirmed=True, conf={"evil": 1})["status"] == "invalid"


def test_k8s_unauthorized_modifications_are_rejected(monkeypatch):
    def boom():
        raise AssertionError("must not reach the cluster")

    monkeypatch.setattr(k8, "_apps", boom)
    assert k8.restart_deployment(namespace="default", confirmed=True)["status"] == "denied"
    assert k8.restart_deployment(confirmed=False)["status"] == "denied"
    assert k8.restart_deployment(deployment="other", confirmed=True)["status"] == "denied"


def test_dvc_rejects_unapproved_commands_and_bad_refs():
    with pytest.raises(PermissionError):
        dv._run(["rm", "-rf", "/"])
    assert dv.compare_versions("--bad", "main")["status"] == "invalid"


def test_compare_experiments_needs_two_names():
    assert ml.compare_experiments(["only-one"])["status"] == "invalid"
