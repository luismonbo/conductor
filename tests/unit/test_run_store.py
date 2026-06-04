from __future__ import annotations

from collections.abc import AsyncGenerator

import aiosqlite
import pytest

from harness.observability.run_store import RunStore
from harness.observability.token_accumulator import TokenAccumulator


@pytest.fixture
async def store() -> AsyncGenerator[tuple[RunStore, aiosqlite.Connection], None]:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    s = RunStore(conn)
    await s.create_table()
    yield s, conn
    await conn.close()


async def test_start_and_finish_run(store):
    s, conn = store
    acc = TokenAccumulator()
    acc.add(input_tokens=100, output_tokens=50)

    await s.start_run("run-1", "thread-1", "default", "fake")
    await s.finish_run("run-1", acc, "final_answer")

    async with conn.execute("SELECT * FROM runs WHERE run_id = 'run-1'") as cur:
        row = await cur.fetchone()

    assert row is not None
    assert row["thread_id"] == "thread-1"
    assert row["agent"] == "default"
    assert row["backend"] == "fake"
    assert row["stopped_reason"] == "final_answer"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["iterations"] == 1
    assert row["started_at"] is not None
    assert row["finished_at"] is not None


async def test_create_table_is_idempotent(store):
    s, _ = store
    await s.create_table()  # second call should not raise
    await s.create_table()


async def test_finish_run_unknown_id_logs_warning(store, caplog):
    import logging
    s, _ = store
    acc = TokenAccumulator()
    with caplog.at_level(logging.WARNING, logger="harness.observability.run_store"):
        await s.finish_run("nonexistent-id", acc, "error")
    assert "no row updated" in caplog.text


async def test_start_run_records_started_at(store):
    s, conn = store
    await s.start_run("run-2", "thread-2", "default", "fake")

    async with conn.execute("SELECT started_at, finished_at FROM runs WHERE run_id = 'run-2'") as cur:
        row = await cur.fetchone()

    assert row["started_at"] is not None
    assert row["finished_at"] is None
