"""FastAPI entry point.

POST /chat         — legacy blocking endpoint (unchanged; uses ReActAgent).
POST /chat/stream  — LangGraph streaming endpoint; emits SSE AgentEvents.
POST /resume/{thread_id} — resume a paused (interrupted) graph run.
POST /cancel/{thread_id} — cancel a running streaming task.
GET  /health       — liveness check.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command

import aiosqlite

from harness.adapters.memory.in_memory import InMemoryShortTerm
from harness.api.schemas import ChatRequest, ChatResponse, ResumeRequest
from harness.config.settings import get_settings
from harness.core.types import AgentEvent, AgentState, Message, Role
from harness.observability.run_store import RunStore
from harness.observability.token_accumulator import TokenAccumulator
from harness.observability.tracer import TraceCollector
from harness.orchestration.build import build_agent, build_agent_registry
from harness.orchestration.checkpointer import build_checkpointer

# Lazy-initialized module-level state
_run_store: RunStore | None = None
_run_store_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _run_store
    if _run_store is not None:
        await _run_store._conn.close()


app = FastAPI(title="Agent Harness", lifespan=lifespan)

_cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

_short_term = InMemoryShortTerm()
_running: dict[str, asyncio.Task] = {}

# Lazy-initialized agent registry (shared checkpointer keeps state across requests)
_registry: dict[str, object] | None = None


def _get_run_store_lock() -> asyncio.Lock:
    global _run_store_lock
    if _run_store_lock is None:
        _run_store_lock = asyncio.Lock()
    return _run_store_lock


async def _get_run_store() -> RunStore | None:
    global _run_store
    if _run_store is not None:
        return _run_store
    settings = get_settings()
    if settings.checkpointer == "memory":
        return None
    async with _get_run_store_lock():
        if _run_store is None:
            conn = await aiosqlite.connect(settings.checkpointer_url)
            _run_store = RunStore(conn)
            await _run_store.create_table()
    return _run_store


async def _get_registry() -> dict[str, object]:
    global _registry
    if _registry is None:
        settings = get_settings()
        cp = await build_checkpointer(settings)
        _registry = build_agent_registry(settings, cp)
    return _registry


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": get_settings().llm_backend}


# ---------------------------------------------------------------------------
# Legacy blocking endpoint — untouched
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    conversation_id = req.thread_id or str(uuid.uuid4())

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


# ---------------------------------------------------------------------------
# Shared SSE generator
# ---------------------------------------------------------------------------

def _make_sse_generator(thread_id: str, event_queue: asyncio.Queue, task: asyncio.Task):
    async def _sse():
        try:
            yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(dataclasses.asdict(item))}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return _sse()


async def _run_graph(
    graph,
    invoke_arg,
    config: dict,
    thread_id: str,
    run_id: str,
    event_queue: asyncio.Queue,
    accumulator: TokenAccumulator,
    stopped_reason_holder: list[str],
    run_store: RunStore | None,
) -> None:
    try:
        await graph.ainvoke(invoke_arg, config)
    except asyncio.CancelledError:
        stopped_reason_holder[0] = "cancelled"
        await event_queue.put(AgentEvent(type="error", text="cancelled"))
        raise
    except Exception as exc:
        stopped_reason_holder[0] = "error"
        await event_queue.put(AgentEvent(type="error", text=str(exc)))
    finally:
        _running.pop(thread_id, None)
        if run_store:
            # Shield finish_run so it completes even if the task is being cancelled
            # (the SSE generator's finally block calls task.cancel() once it sees None).
            # finish_run must land before we send the sentinel so the task is already
            # done by the time the generator can cancel it.
            try:
                await asyncio.shield(
                    run_store.finish_run(run_id, accumulator, stopped_reason_holder[0])
                )
            except asyncio.CancelledError:
                pass
        await event_queue.put(None)


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream agent events as Server-Sent Events.

    First SSE frame: ``{"thread_id": "<uuid>"}``
    Subsequent frames: each AgentEvent serialized with dataclasses.asdict().
    Stream ends after a ``final``, ``interrupt``, or ``error`` event.
    """
    settings = get_settings()
    thread_id = req.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    agent_name = req.agent or settings.agent

    registry = await _get_registry()
    graph = registry.get(agent_name) or registry[settings.agent]

    accumulator = TokenAccumulator()
    stopped_reason_holder: list[str] = ["unknown"]
    event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "event_queue": event_queue,
            "token_accumulator": accumulator,
            "stopped_reason_holder": stopped_reason_holder,
        }
    }
    input_state = {
        "messages": [Message(role=Role.USER, content=req.message)],
        "iteration": 0,
        "max_iterations": settings.max_iterations,
    }

    run_store = await _get_run_store()
    if run_store:
        await run_store.start_run(run_id, thread_id, agent_name, settings.llm_backend)

    task = asyncio.create_task(_run_graph(
        graph, input_state, config, thread_id, run_id,
        event_queue, accumulator, stopped_reason_holder, run_store,
    ))
    _running[thread_id] = task

    return StreamingResponse(
        _make_sse_generator(thread_id, event_queue, task),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Resume endpoint
# ---------------------------------------------------------------------------

@app.post("/resume/{thread_id}")
async def resume_run(thread_id: str, req: ResumeRequest) -> StreamingResponse:
    """Resume a paused graph run (after an interrupt).

    Response: same SSE stream as /chat/stream (starts with thread_id frame).
    """
    settings = get_settings()
    registry = await _get_registry()
    graph = registry[settings.agent]
    run_id = str(uuid.uuid4())

    accumulator = TokenAccumulator()
    stopped_reason_holder: list[str] = ["unknown"]
    event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "event_queue": event_queue,
            "token_accumulator": accumulator,
            "stopped_reason_holder": stopped_reason_holder,
        }
    }

    run_store = await _get_run_store()
    if run_store:
        await run_store.start_run(run_id, thread_id, settings.agent, settings.llm_backend)

    task = asyncio.create_task(_run_graph(
        graph, Command(resume=req.decision), config, thread_id, run_id,
        event_queue, accumulator, stopped_reason_holder, run_store,
    ))
    _running[thread_id] = task

    return StreamingResponse(
        _make_sse_generator(thread_id, event_queue, task),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Cancel endpoint
# ---------------------------------------------------------------------------

@app.post("/cancel/{thread_id}")
async def cancel_run(thread_id: str) -> dict:
    """Cancel a running streaming agent task by thread_id."""
    task = _running.get(thread_id)
    if task and not task.done():
        task.cancel()
        return {"status": "cancelled", "thread_id": thread_id}
    return {"status": "not_found", "thread_id": thread_id}
