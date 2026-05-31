"""FastAPI entry. POST /chat runs one agent turn with short-term history.

Translates DTOs <-> core types at the boundary, runs the agent, persists the
turn to short-term memory, and returns the trace summary so observability is
visible from the first request.

POST /chat/stream streams agent events as SSE until the agent finishes.
POST /cancel/{conversation_id} cancels a running streaming task.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from harness.adapters.memory.in_memory import InMemoryShortTerm
from harness.api.schemas import ChatRequest, ChatResponse
from harness.config.settings import get_settings
from harness.core.types import AgentEvent, AgentState, Message, Role
from harness.observability.tracer import StreamingTracer, TraceCollector
from harness.orchestration.build import build_agent

app = FastAPI(title="Agent Harness")
_short_term = InMemoryShortTerm()

# Map of conversation_id -> running asyncio.Task for streaming runs.
_running: dict[str, asyncio.Task] = {}


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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream agent events as Server-Sent Events.

    First SSE line: ``{"conversation_id": "<id>"}``
    Subsequent lines: each AgentEvent serialized with dataclasses.asdict().
    Stream ends after a ``done`` or ``error`` event.
    """
    settings = get_settings()
    conversation_id = req.conversation_id or str(uuid.uuid4())

    user_msg = Message(role=Role.USER, content=req.message)
    await _short_term.append(conversation_id, user_msg)
    history = await _short_term.history(conversation_id)

    tracer = StreamingTracer()
    agent = build_agent(settings, tracer=tracer)
    state = AgentState(messages=list(history), max_iterations=settings.max_iterations)

    async def _run() -> None:
        try:
            result = await agent.run(state)
            await _short_term.append(
                conversation_id,
                Message(role=Role.ASSISTANT, content=result.output),
            )
            await tracer.finish(
                AgentEvent(
                    type="done",
                    text=result.output,
                    stopped_reason=result.stopped_reason,
                )
            )
        except asyncio.CancelledError:
            await tracer.finish(AgentEvent(type="error", text="cancelled"))
            raise
        except Exception as exc:
            await tracer.finish(AgentEvent(type="error", text=str(exc)))
        finally:
            _running.pop(conversation_id, None)

    task = asyncio.create_task(_run())
    _running[conversation_id] = task

    async def _sse():
        yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
        async for event in tracer.drain():
            yield f"data: {json.dumps(dataclasses.asdict(event))}\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/cancel/{conversation_id}")
async def cancel_run(conversation_id: str) -> dict:
    """Cancel a running streaming agent task by conversation_id."""
    task = _running.get(conversation_id)
    if task and not task.done():
        task.cancel()
        return {"status": "cancelled", "conversation_id": conversation_id}
    return {"status": "not_found", "conversation_id": conversation_id}
