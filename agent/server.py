"""HTTP interface of the agent: POST /chat."""
import logging

from fastapi import FastAPI

from agent.orchestrator import Agent
from agent.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="mlops-copilot agent")
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    return get_agent().run(req.message, confirm_actions=req.confirm_actions)
