"""Wire DTOs. Deliberately separate from core.types so the HTTP contract can
evolve independently of the domain model."""
from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    output: str
    conversation_id: str
    stopped_reason: str
    trace_summary: dict


class AgentEventDTO(BaseModel):
    type: str
    text: str = ""
    name: str = ""
    args: dict = {}
    call_id: str = ""
    is_error: bool = False
    stopped_reason: str = ""
