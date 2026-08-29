"""FastAPI entry point.

POST /chat/stream  — LangGraph streaming endpoint; emits SSE AgentEvents.
POST /resume/{thread_id} — resume a paused (interrupted) graph run.
POST /cancel/{thread_id} — cancel a running streaming task.
GET  /health       — liveness check.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.types import Command
from slowapi.errors import RateLimitExceeded

import aiosqlite

from harness.api.auth import require_api_key
from harness.api.rate_limit import default_limit, limiter, strict_limit
from harness.api.schemas import ChatRequest, ResumeRequest
from harness.config.settings import get_settings
from harness.core.types import AgentEvent, Message, Role
from harness.observability.run_store import RunStore
from harness.observability.token_accumulator import TokenAccumulator
from harness.orchestration.build import build_agent_registry
from harness.orchestration.checkpointer import build_checkpointer

# Settings() reads HARNESS_-prefixed vars straight from .env itself, but this
# module's own os.environ.get() calls (Langfuse keys in _build_callbacks)
# need the file loaded into the real process environment first — Docker's
# env_file: only does this for the containerized api service, not `make api`.
# override=False: a real shell export always wins over .env.
load_dotenv(override=False)

# Lazy-initialized module-level state
_run_store: RunStore | None = None
_run_store_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate settings at boot: fail fast here, not on the first request.
    get_settings()
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
    expose_headers=["Retry-After"],
    allow_credentials=False,
)

app.state.limiter = limiter


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 body reshaped to this API's {"detail": ...} convention (matching the
    404 on GET /threads/{id} and other existing error responses), rather than
    slowapi's own default body shape.

    Retry-After is hardcoded to 60s: both configured tiers (rate_limit_strict,
    rate_limit_default) are per-minute by design -- see
    docs/superpowers/specs/2026-08-27-rate-limiting-design.md. Revisit if a
    non-minute granularity is ever introduced.
    """
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down and try again shortly."},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

_running: dict[str, asyncio.Task] = {}

# Backends where a client-supplied ChatRequest.model is safe to honor:
# openai_compatible routes per-model through a proxy by design; fake never
# calls anything real. A direct-credentialed backend (azure) has one fixed
# real deployment configured -- honoring an arbitrary override would
# silently redirect requests to a different deployment under the same
# credentials. Allowlist, not denylist, so a future direct-credentialed
# backend is safe (ignored) by default rather than needing to remember to
# add it here. See docs/devlog/010.
_MODEL_OVERRIDE_SAFE_BACKENDS = {"openai_compatible", "fake"}

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
# Model listing
# ---------------------------------------------------------------------------

async def _list_model_ids(client: Any) -> list[str]:
    page = await client.models.list()
    return sorted(m.id for m in page.data)


@app.get("/models")
@limiter.limit(default_limit)
async def list_models(request: Request, _auth: None = Depends(require_api_key)) -> dict:
    """Model profiles the proxy serves; the UI picker feeds from this.

    default=None (not settings.default_model/llm_model) whenever there's no
    real picker: the frontend sends `default` straight back as a live
    per-call model override (ChatRequest.model -> model_override), and a
    direct backend (fake, azure) has no named profiles for that override to
    mean anything — it would silently replace the real deployment/model
    (e.g. an Azure deployment name) with whatever local-dev value happens to
    sit in HARNESS_DEFAULT_MODEL/HARNESS_LLM_MODEL.
    """
    settings = get_settings()
    if settings.llm_backend != "openai_compatible":
        return {"models": [], "default": None}
    default = settings.default_model or settings.llm_model
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "sk-no-key-required",
    )
    try:
        return {"models": await _list_model_ids(client), "default": default}
    except Exception:
        # Proxy down is not an API error: the picker just hides.
        return {"models": [], "default": default}


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

async def _thread_state_messages(graph: Any, thread_id: str) -> list[Message] | None:
    snapshot = await graph.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    if not snapshot or not snapshot.values:
        return None
    return snapshot.values.get("messages") or None


def _title_from(messages: list[Message]) -> str:
    first_user = next((m for m in messages if m.role == Role.USER), None)
    return (first_user.content[:60] if first_user else "")


@app.get("/threads")
@limiter.limit(default_limit)
async def list_threads(request: Request, _auth: None = Depends(require_api_key)) -> dict:
    run_store = await _get_run_store()
    if run_store is None:
        return {"threads": []}
    settings = get_settings()
    registry = await _get_registry()
    graph = registry[settings.agent]
    rows = await run_store.list_threads()

    async def _summarize(row: dict) -> dict:
        try:
            messages = await _thread_state_messages(graph, row["thread_id"])
            title = _title_from(messages) if messages else ""
        except Exception:
            title = ""
        return {**row, "title": title}

    threads = await asyncio.gather(*(_summarize(r) for r in rows))
    return {"threads": list(threads)}


@app.get("/threads/{thread_id}")
@limiter.limit(default_limit)
async def thread_messages(request: Request, thread_id: str, _auth: None = Depends(require_api_key)) -> dict:
    settings = get_settings()
    registry = await _get_registry()
    graph = registry[settings.agent]
    messages = await _thread_state_messages(graph, thread_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="unknown thread")
    out = [
        {
            "role": m.role.value,
            "content": m.content,
            "tool_calls": [
                {"name": tc.name, "args": tc.arguments, "call_id": tc.id}
                for tc in (m.tool_calls or ())
            ],
            "tool_call_id": m.tool_call_id or "",
            "name": m.name or "",
        }
        for m in messages
        if m.role != Role.SYSTEM
    ]
    return {"thread_id": thread_id, "messages": out}


# ---------------------------------------------------------------------------
# Observability + error shaping
# ---------------------------------------------------------------------------

def _make_langfuse_handler() -> Any:
    # Separate function so tests can stub construction failures.
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def _build_callbacks(thread_id: str, agent_name: str) -> tuple[list, dict]:
    """Langfuse tracing config. Fire-and-forget: any failure returns empty.

    The SDK reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    from the environment itself; we only gate on their presence.
    """
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return [], {}
    try:
        handler = _make_langfuse_handler()
    except Exception:
        logging.getLogger(__name__).exception("langfuse handler unavailable")
        return [], {}
    return [handler], {
        "langfuse_session_id": thread_id,
        "langfuse_tags": [agent_name],
    }


_CONNECTION_ERROR_NAMES = {"APIConnectionError", "APITimeoutError", "ConnectError"}


def _friendly_error(exc: Exception) -> str:
    if type(exc).__name__ in _CONNECTION_ERROR_NAMES:
        return (
            "LLM gateway unreachable — is the stack running? Try `make up`. "
            f"({exc})"
        )
    return str(exc)


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
        result = await graph.ainvoke(invoke_arg, config)
        # Surface any interrupt payloads AFTER ainvoke returns so that resume
        # reruns of the same node don't fire a duplicate interrupt event into
        # the new SSE stream.
        for intr in (result or {}).get("__interrupt__", ()):
            stopped_reason_holder[0] = "interrupted"
            await event_queue.put(AgentEvent(type="interrupt", args=intr.value))
    except asyncio.CancelledError:
        stopped_reason_holder[0] = "cancelled"
        await event_queue.put(AgentEvent(type="error", text="cancelled"))
        raise
    except Exception as exc:
        stopped_reason_holder[0] = "error"
        await event_queue.put(AgentEvent(type="error", text=_friendly_error(exc)))
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
@limiter.limit(strict_limit)
async def chat_stream(request: Request, req: ChatRequest, _auth: None = Depends(require_api_key)) -> StreamingResponse:
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
    callbacks, lf_metadata = _build_callbacks(thread_id, agent_name)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "event_queue": event_queue,
            "token_accumulator": accumulator,
            "stopped_reason_holder": stopped_reason_holder,
        },
        "callbacks": callbacks,
        "metadata": lf_metadata,
    }
    input_state = {
        "messages": [Message(role=Role.USER, content=req.message)],
        "iteration": 0,
        "max_iterations": settings.max_iterations,
        "model_override": req.model if settings.llm_backend in _MODEL_OVERRIDE_SAFE_BACKENDS else None,
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
@limiter.limit(strict_limit)
async def resume_run(request: Request, thread_id: str, req: ResumeRequest, _auth: None = Depends(require_api_key)) -> StreamingResponse:
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
    callbacks, lf_metadata = _build_callbacks(thread_id, settings.agent)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "event_queue": event_queue,
            "token_accumulator": accumulator,
            "stopped_reason_holder": stopped_reason_holder,
        },
        "callbacks": callbacks,
        "metadata": lf_metadata,
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
@limiter.limit(default_limit)
async def cancel_run(request: Request, thread_id: str, _auth: None = Depends(require_api_key)) -> dict:
    """Cancel a running streaming agent task by thread_id."""
    task = _running.get(thread_id)
    if task and not task.done():
        task.cancel()
        return {"status": "cancelled", "thread_id": thread_id}
    return {"status": "not_found", "thread_id": thread_id}
