"""Wire DTOs. Deliberately separate from core.types so the HTTP contract can
evolve independently of the domain model."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None   # generated server-side if absent
    model: str | None = None       # UI picker override; pins beat it


class ResumeRequest(BaseModel):
    # The interrupt this answers: its frame's args.interrupt_id. Required, since an
    # answer that names none could only be applied to whichever card is pending.
    interrupt_id: str
    decision: dict[str, Any]       # {"action": "approve" | "deny" | "feedback", ...} | {"approved": bool}


class AgentEventDTO(BaseModel):
    type: str
    text: str = ""
    name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""
    is_error: bool = False
    stopped_reason: str = ""
