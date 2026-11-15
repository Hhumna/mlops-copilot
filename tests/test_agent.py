import json
import logging
from types import SimpleNamespace as NS

from conftest import call, reply

import agent.orchestrator as orch


class _Rate(Exception):
    pass


def _patch_llm(monkeypatch, good_keys):
    class Completions:
        def __init__(self, key):
            self.key = key

        def create(self, **kw):
            if self.key not in good_keys:
                raise _Rate("429")
            return reply("hello")

    class FakeOpenAI:
        def __init__(self, api_key, **kw):
            self.chat = NS(completions=Completions(api_key))

    monkeypatch.setattr(orch, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(orch, "RateLimitError", _Rate)


def test_selects_and_executes_tool(make_agent, fake_tool):
    fake_tool("find_best_run", lambda **kw: {"status": "ok", "data": {"metric": kw["metric"]}})
    agent = make_agent([
        reply(calls=[call("find_best_run", '{"experiment_name": "e", "metric": "f1"}')]),
        reply("done"),
    ])
    out = agent.run("best run?")
    assert out["status"] == "ok" and out["answer"] == "done"
    assert [(t["tool"], t["status"]) for t in out["tools_used"]] == [("find_best_run", "ok")]


def test_unknown_tool_and_bad_arguments_are_rejected(make_agent):
    agent = make_agent([
        reply(calls=[call("drop_database", "{}", "a"), call("list_pods", "not json", "b")]),
        reply("ok"),
    ])
    out = agent.run("x")
    assert [t["status"] for t in out["tools_used"]] == ["invalid", "invalid"]


def test_tool_failure_is_passed_to_model_not_hidden(make_agent, fake_tool):
    fake_tool("list_experiments", lambda **kw: {"status": "error", "error": "connection refused"})
    seen = []
    agent = make_agent([reply(calls=[call("list_experiments")]), reply("MLflow is unreachable.")], seen)
    agent.run("what experiments exist?")
    last = seen[1][-1]
    assert last["role"] == "tool" and "connection refused" in last["content"]


def test_model_cannot_confirm_mutations_itself(make_agent, fake_tool):
    received = {}
    fake_tool("restart_deployment", lambda **kw: received.update(kw) or {"status": "denied"})
    agent = make_agent([
        reply(calls=[call("restart_deployment", '{"confirmed": true}')]),
        reply("needs confirmation"),
    ])
    agent.run("restart it", confirm_actions=False)
    assert received["confirmed"] is False


def test_retraining_question_forces_airflow_and_registry_checks(make_agent, fake_tool):
    names = ["get_dag_runs", "find_failed_tasks", "get_model_versions", "find_unhealthy_pods", "get_experiment_dataset"]
    for n in names:
        fake_tool(n, lambda **kw: {"status": "ok", "data": {}})
    agent = make_agent([
        reply("No retraining needed."),
        reply(calls=[call(n, "{}", str(i)) for i, n in enumerate(names)]),
        reply("Evidence checked."),
    ])
    out = agent.run("Is retraining necessary?")
    assert {t["tool"] for t in out["tools_used"]} == set(names)
    assert out["answer"] == "Evidence checked."


def test_rotates_to_next_key_when_rate_limited(make_agent, monkeypatch):
    monkeypatch.setenv("LLM_API_KEYS", "k1,k2")
    _patch_llm(monkeypatch, good_keys={"k2"})
    agent = orch.Agent()
    out = agent.run("hi")
    assert out["status"] == "ok" and agent.idx == 1


def test_all_keys_exhausted_reports_unavailable_without_inventing(make_agent, monkeypatch):
    monkeypatch.setenv("LLM_API_KEYS", "k1,k2")
    _patch_llm(monkeypatch, good_keys=set())
    out = orch.Agent().run("how many pods are running?")
    assert out["status"] == "llm_unavailable" and out["tools_used"] == []
    assert "No operational results were guessed" in out["answer"]


def test_structured_logs_have_required_fields_and_no_secrets(make_agent, fake_tool, caplog):
    fake_tool("list_pods", lambda **kw: {"status": "ok", "data": []})
    caplog.set_level(logging.INFO, logger="agent.orchestrator")
    out = make_agent([reply(calls=[call("list_pods")]), reply("fine")]).run("pods?")
    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "agent.orchestrator"]
    tool_events = [e for e in events if e["event"] == "tool_call"]
    assert tool_events and {"request_id", "tool", "status", "duration_ms"} <= tool_events[0].keys()
    assert tool_events[0]["request_id"] == out["request_id"]
    assert "secret-test-key" not in caplog.text


def test_tool_routing_limits_schemas_but_retraining_gets_all():
    k8s_only = orch._select("Why is my prediction API unavailable? Check pods and logs.")
    assert k8s_only and all(orch.GROUP[n] == "kubernetes" for n in k8s_only)
    assert set(orch._select("Is retraining necessary?")) == set(orch.TOOLS)
