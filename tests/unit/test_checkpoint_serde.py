"""What LangGraph may load back from a checkpoint.

LangGraph's default loads any type a checkpoint names, so write access to the checkpoint
tables becomes code execution on the next load (CVE-2026-28277). Strict mode
(LANGGRAPH_STRICT_MSGPACK=true: the Dockerfile sets it, conftest sets it for the suite)
loads only LangGraph's built-in safe types plus an allowlist. A compiled graph derives
its own from the state schema; open_storage's savers carry one as well, for reads outside
a graph and for every run where the variable is unset (make api, scripts, evaluation).
A blocked type comes back as plain data instead of raising, so only a test notices.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

import harness.adapters.postgres.pool as pool_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.tools.calculator import CalculatorTool
from harness.config.settings import Settings
from harness.core.tools.registry import ToolRegistry
from harness.core.types import CHECKPOINT_TYPES, LLMResponse, Message, Role, ToolCall
from harness.observability.run_store.postgres import PostgresRunStore
from harness.orchestration.loop import build_graph
from harness.orchestration.storage import open_storage

_CALL = ToolCall(id="c1", name="calculator", arguments={"expression": "3+3"})


@dataclass(frozen=True)
class _Unlisted:
    n: int


class _ApprovedCalculator(CalculatorTool):
    @property
    def requires_approval(self) -> bool:
        return True


@pytest.fixture
def serde_events() -> Iterator[list[dict]]:
    """Every allow or block decision the serializer makes while a test runs.

    Its log warning appears once per process, so it can't show whether this test's
    loads were allowed; these events fire every time."""
    events: list[dict] = []
    unregister = register_serde_event_listener(events.append)
    yield events
    unregister()


def _settings(backend: str, tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None, api_key="test-key", input_guard_enabled=False,
        checkpointer=backend, checkpointer_url=str(tmp_path / "h.sqlite"),
    )


def _graph(checkpointer, replies: list[LLMResponse]):
    registry = ToolRegistry()
    registry.register(_ApprovedCalculator())
    return build_graph(
        llm=FakeLLMClient(replies), checkpointer=checkpointer,
        registry=registry, system_prompt="You are a test agent.",
    )


def _config() -> dict:
    return {"configurable": {"thread_id": "t1", "event_queue": asyncio.Queue()}}


async def _pause_at_the_approval_gate(checkpointer) -> None:
    """A thread whose last checkpoint holds a user turn and a tool call awaiting approval."""
    graph = _graph(checkpointer, [LLMResponse(text="", tool_calls=(_CALL,))])
    state = {"messages": [Message(Role.USER, "3+3?")], "iteration": 0, "max_iterations": 5}
    paused = await graph.ainvoke(state, _config())
    assert paused["__interrupt__"], "the thread should be waiting for approval"


def test_the_suite_runs_with_strict_deserialization_on(serde_events):
    serde = JsonPlusSerializer()
    loaded = serde.loads_typed(serde.dumps_typed(_Unlisted(1)))
    assert loaded == {"n": 1}  # the raw data: the type was refused, not rebuilt
    assert [e["kind"] for e in serde_events] == ["msgpack_blocked"]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_the_checkpointer_loads_the_harness_types_outside_a_graph(
    backend, tmp_path, serde_events
):
    storage = await open_storage(_settings(backend, tmp_path))
    try:
        await _pause_at_the_approval_gate(storage.checkpointer)
        saved = await storage.checkpointer.aget_tuple(_config())
    finally:
        await storage.aclose()

    messages = saved.checkpoint["channel_values"]["messages"]
    assert [type(m) for m in messages] == [Message, Message]
    assert {type(m.role) for m in messages} == {Role}
    assert [(m.role, m.content) for m in messages] == [
        (Role.USER, "3+3?"), (Role.ASSISTANT, ""),
    ]
    assert list(messages[-1].tool_calls) == [_CALL]
    assert serde_events == []


async def test_a_paused_thread_resumes_after_a_restart(tmp_path, serde_events):
    settings = _settings("sqlite", tmp_path)
    first = await open_storage(settings)
    try:
        await _pause_at_the_approval_gate(first.checkpointer)
    finally:
        await first.aclose()

    second = await open_storage(settings)
    try:
        graph = _graph(second.checkpointer, [LLMResponse(text="The answer is 6.")])
        finished = await graph.ainvoke(Command(resume={"action": "approve"}), _config())
    finally:
        await second.aclose()

    # The calculator's own result is "6": the last message must be the model's reply.
    last = finished["messages"][-1]
    assert (last.role, last.content) == (Role.ASSISTANT, "The answer is 6.")
    assert serde_events == []


async def test_the_postgres_checkpointer_loads_the_harness_types(monkeypatch, serde_events):
    """The production backend with no database: the pool and the table setup are stubbed,
    so this runs everywhere. tests/integration/test_storage_postgres.py runs the real
    thing wherever a Postgres answers."""

    class _Pool:
        async def close(self) -> None:
            pass

    async def open_pool(dsn: str) -> _Pool:
        return _Pool()

    async def no_setup(self) -> None:
        pass

    monkeypatch.setattr(pool_module, "open_pool", open_pool)
    monkeypatch.setattr(AsyncPostgresSaver, "setup", no_setup)
    monkeypatch.setattr(PostgresRunStore, "create_table", no_setup)
    settings = Settings(
        _env_file=None, api_key="test-key", input_guard_enabled=False,
        checkpointer="postgres",
        checkpointer_url="postgresql://harness:harness@localhost:5432/harness",
    )
    storage = await open_storage(settings)
    try:
        serde = storage.checkpointer.serde
    finally:
        await storage.aclose()

    sent = Message(Role.ASSISTANT, "", tool_calls=(_CALL,))
    got = serde.loads_typed(serde.dumps_typed(sent))
    assert (type(got), type(got.role)) == (Message, Role)
    assert list(got.tool_calls) == [_CALL]
    assert serde_events == []


def test_the_allowlist_names_every_harness_type_the_graph_state_holds():
    """Where the variable is unset (make api, scripts, evaluation) CHECKPOINT_TYPES is
    the only allowlist, even inside the graph. Strict mode, which the image and this suite
    run in, derives the types from GraphState itself and so would hide one missing here."""
    derived = _graph(MemorySaver(), [])._serde_allowlist  # LangGraph's, set only in strict mode
    assert derived is not None, "needs LANGGRAPH_STRICT_MSGPACK=true"
    assert {key for key in derived if key[0].startswith("harness.")} == {
        (cls.__module__, cls.__name__) for cls in CHECKPOINT_TYPES
    }
