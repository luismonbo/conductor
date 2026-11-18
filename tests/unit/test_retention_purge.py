"""purge(): one retention pass over a checkpointer and a run store.

Runs on MemorySaver and an in-memory SqliteRunStore; run_store_clock (tests/conftest.py)
dates the run-store rows, so a thread can be weeks old. The SQLite and Postgres
end-to-end runs are in tests/integration/test_cli_purge.py.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import aiosqlite
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver

from harness.core.guard import GuardDecision
from harness.observability.run_store import RowCounts
from harness.observability.run_store.sqlite import SqliteRunStore
from harness.orchestration.retention import purge

SID = "session-a"
DAY = timedelta(days=1)


class FailingSaver(MemorySaver):
    """A MemorySaver whose adelete_thread fails for the thread ids in `failing`."""

    def __init__(self) -> None:
        super().__init__()
        self.failing: set[str] = set()

    async def adelete_thread(self, thread_id: str) -> None:
        if thread_id in self.failing:
            raise RuntimeError("the checkpoint delete failed")
        await super().adelete_thread(thread_id)


@pytest.fixture
def saver() -> FailingSaver:
    return FailingSaver()


@pytest.fixture
async def store() -> AsyncIterator[SqliteRunStore]:
    conn = await aiosqlite.connect(":memory:")
    opened = SqliteRunStore(conn)
    await opened.create_table()
    yield opened
    await conn.close()


class VisitorWritesMidPass(MemorySaver):
    """While the pass deletes a thread's checkpoints, the visitor's next message lands:
    the API writes its rows, and the graph, which loaded the old state before the
    delete, writes its checkpoint after it."""

    def __init__(self, store: SqliteRunStore) -> None:
        super().__init__()
        self.store = store

    async def adelete_thread(self, thread_id: str) -> None:
        await super().adelete_thread(thread_id)
        await _thread(self, self.store, thread_id)


async def _thread(saver: MemorySaver, store: SqliteRunStore, thread_id: str) -> None:
    """A thread as the API leaves it: a checkpoint, a run and a guard row."""
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    await saver.aput(config, empty_checkpoint(), {}, {})
    run_id = f"run-{uuid.uuid4().hex}"
    await store.start_run(run_id, thread_id, "default", "fake", SID)
    await store.record_guard_decision(
        GuardDecision(allowed=True, stage="classifier"),
        session_id=SID, thread_id=thread_id, run_id=run_id, message="hi",
    )


async def _blocked_first_message(store: SqliteRunStore) -> None:
    await store.record_guard_decision(
        GuardDecision(allowed=False, stage="classifier", reason="off_topic"),
        session_id=SID, thread_id=None, run_id=None, message="hi",
    )


async def _has_checkpoint(saver: MemorySaver, thread_id: str) -> bool:
    return await saver.aget_tuple({"configurable": {"thread_id": thread_id}}) is not None


async def test_a_pass_deletes_stale_threads_and_keeps_recent_ones(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _thread(saver, store, "old")
    run_store_clock.now = t0 + 40 * DAY
    await _thread(saver, store, "new")

    result = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert (result.threads, result.failed_threads, result.dry_run) == (1, 0, False)
    assert result.thread_rows == RowCounts(runs=1, guard_decisions=1)
    assert result.cutoff == t0 + 10 * DAY
    assert not await _has_checkpoint(saver, "old")
    assert await store.owns_thread(SID, "old") is False
    assert await _has_checkpoint(saver, "new")
    assert await store.owns_thread(SID, "new") is True


async def test_unthreaded_rows_go_too(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _blocked_first_message(store)

    result = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert result.unthreaded_rows == RowCounts(runs=0, guard_decisions=1)
    assert await store.count_unthreaded_before(t0 + 40 * DAY) == RowCounts(0, 0)


async def test_a_dry_run_deletes_nothing(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _thread(saver, store, "old")
    await _blocked_first_message(store)
    run_store_clock.now = t0 + 40 * DAY
    await _thread(saver, store, "new")

    result = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30, dry_run=True)

    assert (result.threads, result.failed_threads, result.dry_run) == (1, 0, True)
    assert result.thread_rows == RowCounts(0, 0)
    assert result.unthreaded_rows == RowCounts(runs=0, guard_decisions=1)
    assert await _has_checkpoint(saver, "old")
    assert await store.stale_threads(t0 + 10 * DAY) == ["old"]
    assert await store.count_unthreaded_before(t0 + 10 * DAY) == RowCounts(0, 1)


async def test_a_failed_checkpoint_delete_keeps_the_threads_rows(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _thread(saver, store, "bad")
    await _thread(saver, store, "also-old")
    saver.failing = {"bad"}
    now = t0 + 40 * DAY

    first = await purge(saver, store, now=now, retention_days=30)

    assert (first.threads, first.failed_threads) == (1, 1)
    assert not await _has_checkpoint(saver, "also-old")
    assert await store.owns_thread(SID, "also-old") is False
    # Checkpoints go first, so the failed thread keeps the rows that find it again.
    assert await _has_checkpoint(saver, "bad")
    assert await store.stale_threads(now - 30 * DAY) == ["bad"]

    saver.failing.clear()
    second = await purge(saver, store, now=now, retention_days=30)

    assert (second.threads, second.failed_threads) == (1, 0)
    assert not await _has_checkpoint(saver, "bad")
    assert await store.stale_threads(now - 30 * DAY) == []


async def test_a_message_during_the_pass_keeps_the_thread_findable(store, run_store_clock):
    t0 = run_store_clock.now
    saver = VisitorWritesMidPass(store)
    await _thread(saver, store, "t")
    run_store_clock.now = t0 + 40 * DAY  # the pass, and the visitor's new message

    await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    # The new message's rows survive, so the visitor keeps the conversation and a later
    # pass still reaches its checkpoints: nothing is left that no pass could find.
    assert await store.owns_thread(SID, "t") is True
    assert await _has_checkpoint(saver, "t")
    assert await store.stale_threads(t0 + 100 * DAY) == ["t"]


async def test_a_second_pass_deletes_nothing(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _thread(saver, store, "old")
    await _blocked_first_message(store)
    await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    second = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert (second.threads, second.failed_threads) == (0, 0)
    assert second.thread_rows == RowCounts(0, 0)
    assert second.unthreaded_rows == RowCounts(0, 0)


@pytest.mark.parametrize("retention_days", [0, -5])
async def test_retention_days_below_one_is_refused(saver, store, run_store_clock, retention_days):
    # 0 means "keep everything"; as a cutoff it would be now, which deletes everything.
    t0 = run_store_clock.now
    await _thread(saver, store, "old")

    with pytest.raises(ValueError):
        await purge(saver, store, now=t0 + 40 * DAY, retention_days=retention_days)

    assert await _has_checkpoint(saver, "old")
    assert await store.owns_thread(SID, "old") is True


async def test_now_must_carry_a_time_zone(saver, store):
    with pytest.raises(ValueError):
        await purge(saver, store, now=datetime(2026, 11, 18), retention_days=30)


async def test_a_pass_logs_no_thread_id(saver, store, run_store_clock, caplog):
    # uuid4 ids, as production's are: a word like "old" would match "older" in a log line.
    bad, old = str(uuid.uuid4()), str(uuid.uuid4())
    t0 = run_store_clock.now
    await _thread(saver, store, bad)
    await _thread(saver, store, old)
    saver.failing = {bad}

    with caplog.at_level(logging.DEBUG, logger="harness"):
        await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert bad not in caplog.text
    assert old not in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)
