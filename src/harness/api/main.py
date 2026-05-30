"""FastAPI entry. POST /chat runs one agent turn with short-term history.

Translates DTOs <-> core types at the boundary, runs the agent, persists the
turn to short-term memory, and returns the trace summary so observability is
visible from the first request.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI

from harness.adapters.memory.in_memory import InMemoryShortTerm
from harness.api.schemas import ChatRequest, ChatResponse
from harness.config.settings import get_settings
from harness.core.types import AgentState, Message, Role
from harness.observability.tracer import TraceCollector
from harness.orchestration.build import build_agent

app = FastAPI(title="Agent Harness")
_short_term = InMemoryShortTerm()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": get_settings().llm_backend}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    conversation_id = req.conversation_id or str(uuid.uuid4())

    user_msg = Message(role=Role.USER, content=req.message)
    await _short_term.append(conversation_id, user_msg)

    history = await _short_term.history(conversation_id)

    tracer = TraceCollector()
    agent = build_agent(settings, tracer=tracer)

    state = AgentState(messages=list(history), max_iterations=settings.max_iterations)
    result = await agent.run(state)

    await _short_term.append(
        conversation_id, Message(role=Role.ASSISTANT, content=result.output)
    )

    return ChatResponse(
        output=result.output,
        conversation_id=conversation_id,
        stopped_reason=result.stopped_reason,
        trace_summary=tracer.summary(),
    )
