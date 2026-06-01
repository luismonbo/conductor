"""Wire DTOs. Deliberately separate from core.types so the HTTP contract can
evolve independently of the domain model."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None   # generated server-side if absent
    agent: str = "default"


class ResumeRequest(BaseModel):
    decision: dict[str, Any]       # {"approved": True} | {"approved": False}


class ChatResponse(BaseModel):
    output: str
    conversation_id: str
    stopped_reason: str
    trace_summary: dict


class AgentEventDTO(BaseModel):
    type: str
    text: str = ""
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""
    is_error: bool = False
    stopped_reason: str = ""
