"""open_storage on Postgres: one pool for both, every table before serving, and
threads that outlive the process that wrote them."""
from __future__ import annotations

import asyncio
import logging

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import harness.adapters.postgres.pool as pool_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.postgres.pool import host_of
from harness.config.settings import Settings
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.observability.run_store.postgres import PostgresRunStore
from harness.orchestration.loop import build_graph
from harness.orchestration.storage import open_storage

_THREAD = {"configurable": {"thread_id": "thread-1"}}


def _settings(dsn: str) -> Settings:
    return Settings(
        _env_file=None, api_key="test-key", input_guard_enabled=False,
        checkpointer="postgres", checkpointer_url=dsn,
    )


async def _converse(checkpointer, question: str, reply: str) -> None:
    graph = build_graph(
        llm=FakeLLMClient([LLMResponse(text=reply)]),
        checkpointer=checkpointer,
        registry=ToolRegistry(),
        system_prompt="You are a test agent.",
    )
    state = {"messages": [Message(Role.USER, question)], "iteration": 0, "max_iterations": 2}
    config = {"configurable": {"thread_id": "thread-1", "event_queue": asyncio.Queue()}}
    await graph.ainvoke(state, config)


async def test_both_share_one_pool_and_every_table_exists(postgres_dsn):
    storage = await open_storage(_settings(postgres_dsn))
    try:
        assert isinstance(storage.checkpointer, AsyncPostgresSaver)
        assert isinstance(storage.run_store, PostgresRunStore)
        pool = storage.checkpointer.conn
        assert storage.run_store._pool is pool
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            )
            tables = {row["table_name"] for row in await cur.fetchall()}
        assert {
            "checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations",
            "runs", "guard_decisions",
        } <= tables
    finally:
        await storage.aclose()
    assert pool.closed


async def test_a_thread_outlives_the_storage_that_wrote_it(postgres_dsn):
    """What surviving a restart, or a new revision, means."""
    first = await open_storage(_settings(postgres_dsn))
    try:
        await _converse(first.checkpointer, "hello", "Hi there.")
    finally:
        await first.aclose()

    second = await open_storage(_settings(postgres_dsn))
    try:
        saved = await second.checkpointer.aget_tuple(_THREAD)
        assert saved is not None
        messages = saved.checkpoint["channel_values"]["messages"]
        assert [(m.role, m.content) for m in messages if m.role != Role.SYSTEM] == [
            (Role.USER, "hello"), (Role.ASSISTANT, "Hi there."),
        ]
        await second.checkpointer.adelete_thread("thread-1")  # the retention purge's call (#14)
        assert await second.checkpointer.aget_tuple(_THREAD) is None
    finally:
        await second.aclose()


async def test_a_failure_while_opening_closes_the_pool(postgres_dsn, monkeypatch):
    pools = []
    real_open_pool = pool_module.open_pool

    async def recording_open_pool(dsn):
        pools.append(await real_open_pool(dsn))
        return pools[-1]

    async def failing_create_table(self):
        raise RuntimeError("permission denied for schema public")

    monkeypatch.setattr(pool_module, "open_pool", recording_open_pool)
    monkeypatch.setattr(PostgresRunStore, "create_table", failing_create_table)

    with pytest.raises(RuntimeError, match="permission denied"):
        await open_storage(_settings(postgres_dsn))
    assert pools[0].closed


async def test_it_logs_the_host_never_the_connection_string(postgres_dsn, caplog):
    with caplog.at_level(logging.INFO, logger="harness.orchestration.storage"):
        storage = await open_storage(_settings(postgres_dsn))
    await storage.aclose()
    assert f"storage: postgres on {host_of(postgres_dsn)}" in caplog.text
    assert postgres_dsn not in caplog.text
