from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    confirm_actions: bool = False


class ChatResponse(BaseModel):
    request_id: str
    status: str
    answer: str
    tools_used: list[dict]
