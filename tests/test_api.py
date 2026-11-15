from fastapi.testclient import TestClient

from agent import server


class FakeAgent:
    def run(self, message, confirm_actions=False):
        return {"request_id": "r1", "status": "ok", "answer": f"echo:{message}:{confirm_actions}",
                "tools_used": []}


def test_chat_endpoint_validates_and_forwards(monkeypatch):
    monkeypatch.setattr(server, "get_agent", lambda: FakeAgent())
    c = TestClient(server.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/chat", json={"message": "hi", "confirm_actions": True})
    assert r.status_code == 200 and r.json()["answer"] == "echo:hi:True"
    assert c.post("/chat", json={"message": ""}).status_code == 422
