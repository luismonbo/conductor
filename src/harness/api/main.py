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
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from langgraph.types import Command
from slowapi.errors import RateLimitExceeded

import aiosqlite

from harness.api.auth import require_api_key
from harness.api.rate_limit import default_limit, limiter, strict_limit
from harness.api.session import attach as attach_session_cookie, read_or_mint
from harness.api.schemas import ChatRequest, ResumeRequest
from harness.config.settings import get_settings
from harness.core.types import AgentEvent, Message, Role
from harness.observability.langfuse_tracing import langfuse_configured
from harness.observability.run_store import RunStore
from harness.observability.token_accumulator import TokenAccumulator
from harness.orchestration.build import AgentRuntime, build_input_guard, build_runtime
from harness.security.input_guard import GuardContext, InputGuard

# Settings() reads HARNESS_-prefixed vars straight from .env itself, but this
# module's own os.environ.get() calls (Langfuse keys in _build_callbacks)
# need the file loaded into the real process environment first — Docker's
# env_file: only does this for the containerized api service, not `make api`.
# override=False: a real shell export always wins over .env.
load_dotenv(override=False)

# Lazy-initialized module-level state
_run_store: RunStore | None = None
_run_store_lock: asyncio.Lock | None = None
_runtime: AgentRuntime | None = None
_runtime_lock: asyncio.Lock | None = None
_input_guard: InputGuard | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the entry agent at boot (settings, definition, checkpointer, tools,
    # graph): a missing agent folder or an unavailable required backend stops
    # startup, and /health only answers once the agent can serve.
    await _get_runtime()
    # Build the guard now too, so a malformed TypeSafe key fails the boot.
    _get_input_guard()
    yield
    global _run_store, _runtime, _input_guard
    if _input_guard is not None:
        await _input_guard.aclose()
        _input_guard = None
    if _run_store is not None:
        await _run_store._conn.close()
    if _runtime is not None:
        conn = getattr(_runtime.checkpointer, "conn", None)
        if conn is not None:
            await conn.close()
        _runtime = None


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


@app.middleware("http")
async def _visitor_session(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Attach an opaque visitor id to every request; issue the cookie once.

    Middleware, not a dependency: FastAPI drops a dependency-set Set-Cookie
    header when the endpoint returns a Response directly (see session.py).
    """
    sid, minted = read_or_mint(request)
    request.state.sid = sid
    response = await call_next(request)
    if minted:
        attach_session_cookie(
            response, sid, secure=get_settings().session_cookie_secure
        )
    return response


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

# thread_id -> (session_id of the visitor who started it, task)
_running: dict[str, tuple[str, asyncio.Task]] = {}

# Backends where a client-supplied ChatRequest.model is safe to honor:
# openai_compatible routes per-model through a proxy by design; fake never
# calls anything real. A direct-credentialed backend (azure) has one fixed
# real deployment configured -- honoring an arbitrary override would
# silently redirect requests to a different deployment under the same
# credentials. Allowlist, not denylist, so a future direct-credentialed
# backend is safe (ignored) by default rather than needing to remember to
# add it here. See docs/devlog/010.
_MODEL_OVERRIDE_SAFE_BACKENDS = {"openai_compatible", "fake"}

def _get_run_store_lock() -> asyncio.Lock:
    global _run_store_lock
    if _run_store_lock is None:
        _run_store_lock = asyncio.Lock()
    return _run_store_lock


async def _get_run_store() -> RunStore:
    """The process-wide run store, opened on first use.

    Every checkpointer mode has one, because thread ownership is checked
    against it. Memory mode keeps it in an in-process sqlite database, which
    lives exactly as long as this one connection: the process.

    Cached only once its schema is in place. A store cached before
    create_table() succeeded would lack the session_id column, so every
    continued thread would 404 until restart.
    """
    global _run_store
    if _run_store is not None:
        return _run_store
    settings = get_settings()
    async with _get_run_store_lock():
        if _run_store is None:
            path = (
                ":memory:"
                if settings.checkpointer == "memory"
                else settings.checkpointer_url
            )
            conn = await aiosqlite.connect(path)
            store = RunStore(conn)
            try:
                await store.create_table()
            except BaseException:
                # Not cached, so the next request retries on a fresh connection.
                await conn.close()
                raise
            _run_store = store
        return _run_store


def _get_runtime_lock() -> asyncio.Lock:
    global _runtime_lock
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
    return _runtime_lock


async def _get_runtime() -> AgentRuntime:
    """The entry agent's runtime, built once: at startup by lifespan, or by the
    first request when no lifespan ran (tests). The lock makes concurrent first
    callers share one build instead of each opening a checkpointer."""
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _get_runtime_lock():
        if _runtime is None:
            _runtime = await build_runtime(get_settings())
    return _runtime


def _get_input_guard() -> InputGuard:
    """The process-wide input guard, built on first use (and at boot by lifespan)."""
    global _input_guard
    if _input_guard is None:
        _input_guard = build_input_guard(get_settings())
    return _input_guard


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


async def _require_owned_thread(request: Request, thread_id: str) -> None:
    """404 unless this visitor's session owns thread_id.

    Enforced identically in every checkpointer mode: there is always a run
    store to check against (see _get_run_store). 404 rather than 403, so a
    thread that is not yours is indistinguishable from one that does not exist.
    """
    run_store = await _get_run_store()
    if not await run_store.owns_thread(request.state.sid, thread_id):
        raise HTTPException(status_code=404, detail="unknown thread")


async def _last_assistant_text(graph: Any, thread_id: str) -> str | None:
    """The most recent assistant message with text; tool-call-only turns are skipped."""
    messages = await _thread_state_messages(graph, thread_id) or []
    return next(
        (m.content for m in reversed(messages) if m.role == Role.ASSISTANT and m.content),
        None,
    )


def _utc_midnight_iso() -> str:
    """Start of the current UTC day, in the same format as runs.started_at."""
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


async def _guard_context(
    guard: InputGuard,
    req: ChatRequest,
    runtime: AgentRuntime,
    run_store: RunStore,
) -> GuardContext:
    """Gather what the guard needs, querying only for the checks that are on."""
    limits = guard.limits
    previous_reply = (
        await _last_assistant_text(runtime.graph, req.thread_id)
        if req.thread_id and guard.classifier_enabled
        else None
    )
    thread_turns = (
        await run_store.count_thread_runs(req.thread_id)
        if req.thread_id and limits.max_thread_turns
        else 0
    )
    tokens_today = (
        await run_store.tokens_since(_utc_midnight_iso())
        if limits.daily_token_budget
        else 0
    )
    return GuardContext(
        message=req.message,
        system_prompt=runtime.definition.prompt,
        previous_reply=previous_reply,
        thread_turns=thread_turns,
        tokens_today=tokens_today,
    )


def _blocked_response(refusal: str) -> StreamingResponse:
    """A blocked message's whole reply: one `final` frame and no thread_id frame.

    Without a thread_id frame the frontend keeps the thread id it already has:
    none for a first message, so the visitor's next message starts fresh.
    """
    event = AgentEvent(type="final", text=refusal, stopped_reason="blocked")

    async def _one_frame():
        yield f"data: {json.dumps(dataclasses.asdict(event))}\n\n"

    return StreamingResponse(
        _one_frame(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/threads")
@limiter.limit(default_limit)
async def list_threads(request: Request, _auth: None = Depends(require_api_key)) -> dict:
    run_store = await _get_run_store()
    graph = (await _get_runtime()).graph
    rows = await run_store.list_threads(request.state.sid)

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
    await _require_owned_thread(request, thread_id)
    graph = (await _get_runtime()).graph
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
    if not langfuse_configured():
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

# Provider-side auth/quota/rate-limit failures (e.g. an Azure spend cap or key
# rotation tripping) — never echo these to the public chat widget: the detail
# belongs in the server log (see _run_graph's except block), not the response.
_PROVIDER_UNAVAILABLE_ERROR_NAMES = {
    "RateLimitError",
    "PermissionDeniedError",
    "AuthenticationError",
}


def _friendly_error(exc: Exception) -> str:
    if type(exc).__name__ in _CONNECTION_ERROR_NAMES:
        return (
            "LLM gateway unreachable — is the stack running? Try `make up`. "
            f"({exc})"
        )
    if type(exc).__name__ in _PROVIDER_UNAVAILABLE_ERROR_NAMES:
        return "The assistant is temporarily unavailable. Please try again later."
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
    run_store: RunStore,
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
        logging.getLogger(__name__).exception(
            "agent run failed for thread_id=%s run_id=%s", thread_id, run_id
        )
        stopped_reason_holder[0] = "error"
        await event_queue.put(AgentEvent(type="error", text=_friendly_error(exc)))
    finally:
        _running.pop(thread_id, None)
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

    A message the input guard blocks gets one ``final`` frame with the refusal
    (``stopped_reason: "blocked"``) and nothing else: no thread_id frame, no
    run, no checkpoint write.
    """
    if req.thread_id:
        # Thread ids are server-issued only: a caller may continue a thread
        # this session started, but not name one it doesn't already own.
        # Without this, a guessed/shared thread_id both leaks another
        # visitor's history into this response (it feeds the graph as prior
        # state) and hands this session co-ownership going forward, since
        # start_run below would record it as an owner too.
        await _require_owned_thread(request, req.thread_id)
    runtime = await _get_runtime()
    # Everything below runs, screens and records against the one entry agent
    # built at startup (HARNESS_AGENT), never a client-supplied name.
    settings = runtime.settings
    graph = runtime.graph
    agent_name = settings.agent
    run_store = await _get_run_store()
    thread_id = req.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    guard = _get_input_guard()
    decision = await guard.screen(
        await _guard_context(guard, req, runtime, run_store)
    )
    decision_id = None
    if decision.stage != "none":
        decision_id = await run_store.record_guard_decision(
            decision,
            session_id=request.state.sid,
            thread_id=thread_id if decision.allowed else req.thread_id,
            run_id=run_id if decision.allowed else None,
            message=req.message,
        )
    if not decision.allowed:
        # Never the message text or the sid: those live only in the database.
        logging.getLogger(__name__).info(
            "input guard blocked a message decision_id=%s reason=%s probabilities=%s",
            decision_id, decision.reason, decision.probabilities,
        )
        return _blocked_response(decision.refusal or "")

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
        "run_name": "chat-response",
    }
    input_state = {
        "messages": [Message(role=Role.USER, content=req.message)],
        "iteration": 0,
        "max_iterations": settings.max_iterations,
        "model_override": req.model if settings.llm_backend in _MODEL_OVERRIDE_SAFE_BACKENDS else None,
    }

    await run_store.start_run(
        run_id, thread_id, agent_name, settings.llm_backend, request.state.sid
    )

    task = asyncio.create_task(_run_graph(
        graph, input_state, config, thread_id, run_id,
        event_queue, accumulator, stopped_reason_holder, run_store,
    ))
    _running[thread_id] = (request.state.sid, task)

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
    await _require_owned_thread(request, thread_id)
    runtime = await _get_runtime()
    settings = runtime.settings
    graph = runtime.graph
    # Only a run paused on an interrupt can be resumed. Resuming a thread whose
    # last step was cancelled or failed would re-run the model, outside the
    # input guard's limits.
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.interrupts:
        raise HTTPException(status_code=409, detail="nothing to resume")
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
        "run_name": "chat-resume",
    }

    run_store = await _get_run_store()
    await run_store.start_run(
        run_id, thread_id, settings.agent, settings.llm_backend, request.state.sid
    )

    task = asyncio.create_task(_run_graph(
        graph, Command(resume=req.decision), config, thread_id, run_id,
        event_queue, accumulator, stopped_reason_holder, run_store,
    ))
    _running[thread_id] = (request.state.sid, task)

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
    entry = _running.get(thread_id)
    if entry is not None:
        owner, task = entry
        if owner == request.state.sid and not task.done():
            task.cancel()
            return {"status": "cancelled", "thread_id": thread_id}
    return {"status": "not_found", "thread_id": thread_id}
