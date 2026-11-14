from types import SimpleNamespace as NS

import pytest

import agent.orchestrator as orch


def call(name, args="{}", cid="c1"):
    return NS(id=cid, function=NS(name=name, arguments=args))


def reply(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls or None))])


@pytest.fixture
def make_agent(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.invalid")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("LLM_API_KEYS", raising=False)

    def build(replies, seen=None):
        agent = orch.Agent()
        queue = iter(replies)

        def fake_chat(messages, rid):
            if seen is not None:
                seen.append([dict(m) for m in messages])
            return next(queue)

        monkeypatch.setattr(agent, "_chat", fake_chat)
        return agent

    return build


@pytest.fixture
def fake_tool(monkeypatch):
    def install(name, fn):
        monkeypatch.setitem(orch.TOOLS[name], "fn", fn)

    return install
