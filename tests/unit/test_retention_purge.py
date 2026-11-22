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


class FailingRunStore(SqliteRunStore):
    """A SqliteRunStore whose delete_thread_rows fails for the thread ids in
    `failing_rows`, and whose delete_unthreaded_before fails while `unthreaded_fails`."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        super().__init__(conn)
        self.failing_rows: set[str] = set()
        self.unthreaded_fails = False
        self.unrecorded: list[str] = []

    async def unrecorded_threads(self) -> list[str]:
        # MemorySaver keeps the checkpoints outside this database, so a test names them.
        return list(self.unrecorded)

    async def delete_thread_rows(self, thread_id: str, cutoff: datetime) -> RowCounts:
        if thread_id in self.failing_rows:
            raise RuntimeError("the rows delete failed")
        return await super().delete_thread_rows(thread_id, cutoff)

    async def delete_unthreaded_before(self, cutoff: datetime) -> RowCounts:
        if self.unthreaded_fails:
            raise RuntimeError("the unthreaded delete failed")
        return await super().delete_unthreaded_before(cutoff)


@pytest.fixture
def saver() -> FailingSaver:
    return FailingSaver()


@pytest.fixture
async def store() -> AsyncIterator[FailingRunStore]:
    """Healthy until a test sets failing_rows or unthreaded_fails."""
    conn = await aiosqlite.connect(":memory:")
    opened = FailingRunStore(conn)
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


async def _unrecorded_thread(
    saver: MemorySaver, store: FailingRunStore, thread_id: str, at: datetime
) -> None:
    """A conversation whose two run-store writes both failed: one checkpoint saved at
    `at`, and no rows."""
    checkpoint = empty_checkpoint()
    checkpoint["ts"] = at.isoformat()
    await saver.aput(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}, checkpoint, {}, {}
    )
    store.unrecorded.append(thread_id)


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


async def test_a_failed_rows_delete_is_counted_and_retried(saver, store, run_store_clock, caplog):
    bad, ok = str(uuid.uuid4()), str(uuid.uuid4())
    t0 = run_store_clock.now
    await _thread(saver, store, bad)
    await _thread(saver, store, ok)
    store.failing_rows = {bad}
    now = t0 + 40 * DAY

    with caplog.at_level(logging.DEBUG, logger="harness"):
        first = await purge(saver, store, now=now, retention_days=30)

    assert (first.threads, first.failed_threads) == (1, 1)
    assert first.thread_rows == RowCounts(runs=1, guard_decisions=1)  # the other thread's
    # Its checkpoints went first; its rows stay, so the next pass finds it again.
    assert not await _has_checkpoint(saver, bad)
    assert await store.stale_threads(now - 30 * DAY) == [bad]
    assert "deleting a thread's rows failed (RuntimeError)" in caplog.text
    assert bad not in caplog.text

    store.failing_rows.clear()
    second = await purge(saver, store, now=now, retention_days=30)

    assert (second.threads, second.failed_threads) == (1, 0)
    assert await store.stale_threads(now - 30 * DAY) == []


async def test_a_failed_unthreaded_delete_still_reports_the_threads(
    saver, store, run_store_clock, caplog
):
    t0 = run_store_clock.now
    await _thread(saver, store, "old")
    await _blocked_first_message(store)
    store.unthreaded_fails = True
    now = t0 + 40 * DAY

    with caplog.at_level(logging.DEBUG, logger="harness"):
        result = await purge(saver, store, now=now, retention_days=30)

    # The threads it did delete are still counted, and the pass reports itself incomplete.
    assert (result.threads, result.failed_threads) == (1, 0)
    assert result.thread_rows == RowCounts(runs=1, guard_decisions=1)
    assert result.unthreaded_rows == RowCounts(0, 0)
    assert (result.unthreaded_failed, result.complete) == (True, False)
    assert not await _has_checkpoint(saver, "old")
    assert await store.count_unthreaded_before(now) == RowCounts(0, 1)  # left for the next pass
    assert "deleting the rows with no thread failed (RuntimeError)" in caplog.text


async def test_an_unrecorded_thread_goes_by_its_newest_checkpoint(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _unrecorded_thread(saver, store, "old", t0)
    await _unrecorded_thread(saver, store, "recent", t0 + 35 * DAY)

    result = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert (result.unrecorded_threads, result.threads, result.failed_threads) == (1, 0, 0)
    assert not await _has_checkpoint(saver, "old")
    assert await _has_checkpoint(saver, "recent")


async def test_a_dry_run_counts_unrecorded_threads_and_deletes_none(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _unrecorded_thread(saver, store, "old", t0)

    result = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30, dry_run=True)

    assert result.unrecorded_threads == 1
    assert await _has_checkpoint(saver, "old")


async def test_a_failed_unrecorded_delete_counts_as_a_failed_thread(saver, store, run_store_clock):
    t0 = run_store_clock.now
    await _unrecorded_thread(saver, store, "bad", t0)
    saver.failing = {"bad"}

    result = await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert (result.unrecorded_threads, result.failed_threads, result.complete) == (0, 1, False)
    assert await _has_checkpoint(saver, "bad")


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
    bad, old, lost = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    t0 = run_store_clock.now
    await _thread(saver, store, bad)
    await _thread(saver, store, old)
    await _unrecorded_thread(saver, store, lost, t0)
    saver.failing = {bad, lost}

    with caplog.at_level(logging.DEBUG, logger="harness"):
        await purge(saver, store, now=t0 + 40 * DAY, retention_days=30)

    assert bad not in caplog.text
    assert old not in caplog.text
    assert lost not in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)
