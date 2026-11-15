"""Orchestrator: LLM tool-calling over the DVC, MLflow, Airflow and Kubernetes tools."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from agent.tools import airflow_tools as af
from agent.tools import dvc_tools as dv
from agent.tools import kubernetes_tools as k8
from agent.tools import mlflow_tools as ml

load_dotenv()
log = logging.getLogger("agent.orchestrator")

MAX_STEPS = 14
MAX_TOOL_CHARS = 1800
MUTATING = {"trigger_training", "restart_deployment"}
RETRAIN_REQUIRED = ("get_dag_runs", "find_failed_tasks", "get_model_versions", "find_unhealthy_pods", "get_experiment_dataset")
S, I = {"type": "string"}, {"type": "integer"}

SYSTEM = """You are an MLOps assistant. You can call tools for DVC (dataset versions), \
MLflow (experiments, runs, registry), Airflow (training pipeline) and Kubernetes (inference service).

Rules:
- Every operational claim (metrics, versions, pod state, DAG state) must come from a tool result in this \
conversation. Never answer such things from memory. If a tool returns a status other than "ok", say which \
tool failed and why, and do not guess its result.
- Default MLflow experiment is "mlops-copilot-v3". Default DAG is "training_pipeline". \
Approved Kubernetes namespace is "mlops-app", deployment "inference".
- If the question involves retraining or deployment health, you MUST also call get_dag_runs and find_failed_tasks (Airflow) and get_model_versions (registry) before answering. Use recommend_candidate, not find_best_run, when choosing a model, and report its caveats. Never say something about Airflow or the registry that you did not get from a tool.
- For multi-part questions call all the relevant tools (MLflow, DVC provenance, Kubernetes, Airflow) \
before answering.
- Do not treat the highest metric as proof of production readiness; mention caveats the tools return.
- Never start retraining on your own. Recommend retraining only with evidence: a failed quality gate, \
an explicit freshness policy, or supplied degradation results. Otherwise say it is not justified by the evidence.
- Old warning events for pods that are healthy now are historical; do not present them as the current problem.
- Tools that change state (trigger_training, restart_deployment) are rejected unless the user confirmed. \
If rejected, explain that confirmation is needed.
- Format the final answer with three short parts: Known facts, Uncertainties, Recommended actions.
- Reply in the language the user writes in."""


def _spec(fn, desc, props=None, required=None):
    return {
        "fn": fn,
        "schema": {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                },
            },
        },
    }


_SPECS = [
    # DVC
    _spec(dv.list_datasets, "List DVC-tracked datasets."),
    _spec(dv.get_dataset_version, "Current dataset md5, git commit and git tags."),
    _spec(dv.compare_versions, "Compare dataset at two git refs/tags (e.g. data-v1, data-v2).",
          {"ref_a": S, "ref_b": S}, ["ref_a", "ref_b"]),
    _spec(dv.detect_changes, "Has tracked data changed versus its DVC record?"),
    _spec(dv.explain_reproduce, "Explain how to reproduce a past dataset version (does not execute).",
          {"ref": S}, ["ref"]),
    _spec(dv.get_experiment_dataset, "Dataset version/md5/git commit recorded on an MLflow run.",
          {"run_id": S}, ["run_id"]),
    # MLflow
    _spec(ml.list_experiments, "List MLflow experiments."),
    _spec(ml.list_runs, "Recent runs of an experiment.", {"experiment_name": S, "limit": I},
          ["experiment_name"]),
    _spec(ml.get_run_details, "Params, metrics, tags and artifacts of a run.", {"run_id": S}, ["run_id"]),
    _spec(ml.find_best_run, "Best finished run by a metric.", {"experiment_name": S, "metric": S},
          ["experiment_name"]),
    _spec(ml.compare_runs, "Compare two or more runs.",
          {"run_ids": {"type": "array", "items": S}}, ["run_ids"]),
    _spec(ml.compare_experiments, "Compare experiments by their best finished run on a metric.",
          {"experiment_names": {"type": "array", "items": S}, "metric": S}, ["experiment_names"]),
    _spec(ml.get_model_versions, "Model registry info.", {"model_name": S}),
    _spec(ml.recommend_candidate, "Compare the last N finished runs and pick a candidate with caveats.",
          {"experiment_name": S, "metric": S, "last_n": I}, ["experiment_name"]),
    # Airflow
    _spec(af.list_dags, "List Airflow DAGs."),
    _spec(af.get_dag_runs, "Recent DAG runs and their state.", {"dag_id": S, "limit": I}),
    _spec(af.get_task_instances, "Task states of one DAG run.", {"dag_id": S, "run_id": S},
          ["dag_id", "run_id"]),
    _spec(af.find_failed_tasks, "Failed tasks of a run (default: latest failed run).",
          {"dag_id": S, "run_id": S}),
    _spec(af.get_task_diagnostics, "Error messages and log tail of a task attempt.",
          {"dag_id": S, "run_id": S, "task_id": S, "try_number": I},
          ["dag_id", "run_id", "task_id"]),
    _spec(af.trigger_training, "Trigger the training DAG. Needs user confirmation.",
          {"conf": {"type": "object"}}),
    # Kubernetes
    _spec(k8.list_pods, "List pods in the approved namespace."),
    _spec(k8.list_deployments, "List deployments with desired/ready replicas."),
    _spec(k8.get_pod_logs, "Last log lines of a pod.", {"pod_name": S, "tail": I}, ["pod_name"]),
    _spec(k8.list_services, "Services, ports and ready endpoints."),
    _spec(k8.find_unhealthy_pods, "Unhealthy pods, restarts and recent warning events."),
    _spec(k8.get_deployed_version, "Deployed image tag of the inference deployment."),
    _spec(k8.restart_deployment, "Rolling restart of the inference deployment. Needs user confirmation."),
]
TOOLS = {s["schema"]["function"]["name"]: s for s in _SPECS}

GROUP = {n: s["fn"].__module__.rsplit(".", 1)[-1].removesuffix("_tools") for n, s in TOOLS.items()}
KEYWORDS = {
    "kubernetes": ("pod", "kubernetes", "k8s", "deploy", "api", "service", "unavailable", "health", "log", "restart", "replica"),
    "airflow": ("airflow", "dag", "pipeline", "train", "task", "quality"),
    "mlflow": ("mlflow", "model", "run", "experiment", "metric", "f1", "accuracy", "best", "candidate", "registry"),
    "dvc": ("dataset", "dvc", "data", "version", "provenance", "commit", "reproduce"),
}


def _select(message: str) -> list[str]:
    """Send only relevant tool schemas (keeps requests under the provider's token limit)."""
    m = message.lower()
    if "retrain" in m:
        return list(TOOLS)
    groups = {g for g, words in KEYWORDS.items() if any(w in m for w in words)}
    return [n for n in TOOLS if GROUP[n] in groups] or list(TOOLS)


class LLMUnavailable(RuntimeError):
    pass


def _keys() -> list[str]:
    raw = os.getenv("LLM_API_KEYS") or os.getenv("LLM_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _log(**fields) -> None:
    log.info(json.dumps(fields, default=str))


class Agent:
    def __init__(self) -> None:
        self.keys = _keys()
        if not self.keys:
            raise RuntimeError("No LLM API key configured (LLM_API_KEY / LLM_API_KEYS).")
        self.idx = 0
        self.base_url = os.environ["LLM_BASE_URL"]
        self.model = os.environ["LLM_MODEL"]

    def _chat(self, messages, rid):
        schemas = [TOOLS[n]["schema"] for n in _select(messages[1]["content"])]
        for _ in range(len(self.keys)):
            client = OpenAI(api_key=self.keys[self.idx], base_url=self.base_url,
                            timeout=60, max_retries=0)
            try:
                return client.chat.completions.create(
                    model=self.model, messages=messages, tools=schemas,
                    tool_choice="auto", temperature=0.2,
                )
            except (RateLimitError, AuthenticationError, PermissionDeniedError) as e:
                _log(event="llm_key_rotated", request_id=rid, key_index=self.idx,
                     reason=type(e).__name__)
                self.idx = (self.idx + 1) % len(self.keys)
            except (APIConnectionError, APITimeoutError) as e:
                raise LLMUnavailable("The LLM provider is unreachable.") from e
            except APIStatusError as e:
                if e.status_code == 413:
                    raise LLMUnavailable(
                        "The request is larger than the provider's per-minute token limit."
                    ) from e
                raise
        raise LLMUnavailable("All configured LLM API keys are rate-limited or rejected.")

    def _exec(self, rid, name, raw_args, confirm, used) -> str:
        start = time.time()
        spec = TOOLS.get(name)
        if spec is None:
            out = {"status": "invalid", "error": f"Unknown tool {name!r}"}
        else:
            try:
                args = json.loads(raw_args or "{}")
                if not isinstance(args, dict):
                    raise TypeError("arguments must be an object")
            except (ValueError, TypeError) as e:
                out = {"status": "invalid", "error": f"Bad arguments: {e}"}
            else:
                args.pop("confirmed", None)  # the model can never confirm on the user's behalf
                if name in MUTATING:
                    args["confirmed"] = confirm
                if name == "get_pod_logs":
                    t = args.get("tail")
                    args["tail"] = t if isinstance(t, int) and t <= 15 else 15
                try:
                    out = spec["fn"](**args)
                except TypeError as e:
                    out = {"status": "invalid", "error": f"Bad arguments: {e}"}
        ms = int((time.time() - start) * 1000)
        _log(event="tool_call", request_id=rid, tool=name, status=out.get("status"), duration_ms=ms)
        used.append({"tool": name, "status": out.get("status"), "duration_ms": ms})
        text = json.dumps(out, default=str, separators=(",", ":"))
        return text[:MAX_TOOL_CHARS] + (" ...[truncated]" if len(text) > MAX_TOOL_CHARS else "")

    @staticmethod
    def _missing_tools(message: str, used: list[dict]) -> list[str]:
        if "retrain" not in message.lower():
            return []
        done = {u["tool"] for u in used}
        return [t for t in RETRAIN_REQUIRED if t not in done]

    def run(self, message: str, confirm_actions: bool = False) -> dict:
        rid = uuid.uuid4().hex[:12]
        system = SYSTEM
        if confirm_actions:
            system += "\nThe user has explicitly confirmed state-changing actions for this request. If they asked for one, call the tool now instead of asking again."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
        used: list[dict] = []
        nudged = False
        _log(event="request", request_id=rid)
        try:
            for _ in range(MAX_STEPS):
                m = self._chat(messages, rid).choices[0].message
                if not m.tool_calls:
                    missing = [] if nudged else self._missing_tools(message, used)
                    if missing:
                        nudged = True
                        messages.append({"role": "assistant", "content": m.content or ""})
                        messages.append({"role": "user", "content": "Before answering you must also call: " + ", ".join(missing) + ". Call them now, then give the final answer."})
                        continue
                    return {"request_id": rid, "status": "ok", "answer": m.content or "",
                            "tools_used": used}
                messages.append({
                    "role": "assistant", "content": m.content,
                    "tool_calls": [
                        {"id": t.id, "type": "function",
                         "function": {"name": t.function.name, "arguments": t.function.arguments}}
                        for t in m.tool_calls
                    ],
                })
                for t in m.tool_calls:
                    result = self._exec(rid, t.function.name, t.function.arguments,
                                        confirm_actions, used)
                    messages.append({"role": "tool", "tool_call_id": t.id, "content": result})
            return {"request_id": rid, "status": "incomplete", "tools_used": used,
                    "answer": "Stopped after too many tool steps; results so far are in tools_used."}
        except LLMUnavailable as e:
            _log(event="llm_unavailable", request_id=rid, error=str(e))
            return {"request_id": rid, "status": "llm_unavailable", "tools_used": used,
                    "answer": f"I cannot answer right now: {e} No operational results were guessed."}