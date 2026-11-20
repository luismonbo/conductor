"""Contract suite for RunStore. Any store implementing the protocol must pass this:
parametrize a new one into `_STORES` and `_open` and it inherits the guarantees, as
tests/contract/test_vector_store.py does for vector stores. What a store writes into
its own columns is pinned in that store's own tests."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from harness.adapters.postgres.pool import open_pool
from harness.core.guard import GuardDecision
from harness.observability.run_store import RowCounts, RunStore
from harness.observability.run_store.postgres import PostgresRunStore
from harness.observability.run_store.sqlite import SqliteRunStore
from harness.observability.token_accumulator import TokenAccumulator
from tests.conftest import postgres_schema

SID_A = "session-a"
SID_B = "session-b"

_STORES = ["sqlite", "postgres"]


async def _open(name: str, stack: AsyncExitStack) -> RunStore:
    """A fresh store with its tables; `stack` closes it after the test."""
    opened: RunStore
    if name == "sqlite":
        conn = await aiosqlite.connect(":memory:")
        stack.push_async_callback(conn.close)
        opened = SqliteRunStore(conn)
    else:
        dsn = stack.enter_context(postgres_schema())  # skips without Postgres
        pool = await open_pool(dsn)
        stack.push_async_callback(pool.close)
        opened = PostgresRunStore(pool)
    await opened.create_table()
    return opened


@pytest.fixture(params=_STORES)
async def store(request) -> AsyncIterator[RunStore]:
    async with AsyncExitStack() as stack:
        yield await _open(request.param, stack)


@pytest.fixture(params=_STORES)
async def broken_store(request) -> AsyncIterator[RunStore]:
    """A store whose every query fails: its connection, or its whole pool, is closed.

    Its one run is finished with tokens, so a healthy store would answer tokens_since
    with 15: anything else is the store's failure, not the answer for a run in flight.
    """
    async with AsyncExitStack() as stack:
        broken = await _open(request.param, stack)
        await broken.start_run("r1", "mine", "default", "fake", SID_A)
        await broken.finish_run("r1", _tokens(10, 5), "final_answer")
    yield broken


def _tokens(input_tokens: int, output_tokens: int) -> TokenAccumulator:
    return TokenAccumulator(input_tokens=input_tokens, output_tokens=output_tokens, iterations=1)


def _decision(**fields) -> GuardDecision:
    return GuardDecision(**{"allowed": False, "stage": "classifier", "reason": "off_topic", **fields})


async def test_create_table_runs_twice_in_a_row(store):
    await store.create_table()
    await store.create_table()


async def test_a_thread_belongs_to_the_session_that_started_it(store):
    await store.start_run("r1", "mine", "default", "fake", SID_A)
    assert await store.owns_thread(SID_A, "mine") is True
    assert await store.owns_thread(SID_B, "mine") is False
    assert await store.owns_thread(SID_A, "never-existed") is False


async def test_list_threads_groups_runs_newest_first(store):
    await store.start_run("r1", "thread-a", "default", "fake", SID_A)
    await store.start_run("r2", "thread-a", "default", "fake", SID_A)
    await store.start_run("r3", "thread-b", "default", "fake", SID_A)
    threads = await store.list_threads(SID_A)
    assert [t["thread_id"] for t in threads] == ["thread-b", "thread-a"]
    assert {t["thread_id"]: t["runs"] for t in threads} == {"thread-a": 2, "thread-b": 1}


async def test_list_threads_gives_last_at_as_utc_iso_8601(store):
    await store.start_run("r1", "t1", "default", "fake", SID_A)
    [thread] = await store.list_threads(SID_A)
    assert isinstance(thread["last_at"], str)
    last_at = datetime.fromisoformat(thread["last_at"])
    assert last_at.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - last_at) < timedelta(minutes=1)


async def test_list_threads_respects_the_limit(store):
    for i in range(5):
        await store.start_run(f"r{i}", f"t{i}", "default", "fake", SID_A)
    assert len(await store.list_threads(SID_A, limit=3)) == 3


async def test_list_threads_shows_only_the_sessions_own_threads(store):
    await store.start_run("r1", "mine", "default", "fake", SID_A)
    await store.start_run("r2", "theirs", "default", "fake", SID_B)
    assert [t["thread_id"] for t in await store.list_threads(SID_A)] == ["mine"]
    assert [t["thread_id"] for t in await store.list_threads(SID_B)] == ["theirs"]


async def test_tokens_since_counts_from_the_cutoff_on(store):
    await store.start_run("r1", "t1", "default", "fake", SID_A)
    await store.finish_run("r1", _tokens(100, 50), "final_answer")
    [thread] = await store.list_threads(SID_A)
    started = datetime.fromisoformat(thread["last_at"])
    assert await store.tokens_since(started - timedelta(days=1)) == 150
    assert await store.tokens_since(started) == 150  # the cutoff itself counts
    assert await store.tokens_since(started + timedelta(microseconds=1)) == 0


async def test_a_run_in_flight_counts_once_it_finishes(store):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    await store.start_run("r1", "t1", "default", "fake", SID_A)
    assert await store.tokens_since(yesterday) == 0
    await store.finish_run("r1", _tokens(10, 5), "final_answer")
    assert await store.tokens_since(yesterday) == 15


async def test_finishing_an_unknown_run_logs_a_warning(store, caplog):
    with caplog.at_level(logging.WARNING, logger="harness.observability.run_store"):
        await store.finish_run("nonexistent-id", _tokens(0, 0), "error")
    assert "no row updated" in caplog.text


async def test_each_guard_decision_gets_its_own_id(store):
    first = await store.record_guard_decision(
        _decision(probabilities={"off_topic": 0.93}),
        session_id=SID_A, thread_id=None, run_id=None, message="write me an essay",
    )
    second = await store.record_guard_decision(
        _decision(allowed=True, reason=None),
        session_id=SID_A, thread_id="t1", run_id="r1", message="hi",
    )
    assert first is not None and second is not None and first != second


async def test_a_message_with_a_nul_character_still_gets_its_decision_row(store):
    # Review Focus 4: Postgres text columns reject NUL; a probe must still leave its row.
    decision_id = await store.record_guard_decision(
        _decision(reason="extraction"),
        session_id=SID_A, thread_id=None, run_id=None, message="ignore\x00 the rules",
    )
    assert decision_id is not None


async def test_a_failing_store_fails_closed_on_ownership_and_limits(broken_store):
    assert await broken_store.owns_thread(SID_A, "mine") is False
    day_ago = datetime.now(timezone.utc) - timedelta(days=1)
    assert await broken_store.tokens_since(day_ago) is None
    assert await broken_store.list_threads(SID_A) == []


async def test_a_failing_store_never_raises_on_writes(broken_store):
    await broken_store.start_run("r2", "t2", "default", "fake", SID_A)
    await broken_store.finish_run("r2", _tokens(1, 1), "final_answer")
    assert await broken_store.record_guard_decision(
        _decision(), session_id=SID_A, thread_id=None, run_id=None, message="hi"
    ) is None


# --- Retention: finding and deleting old rows -------------------------------------
# run_store_clock (tests/conftest.py) dates every row a store writes, so a test can
# write rows that are already weeks old.

DAY = timedelta(days=1)


async def _guard_row(store: RunStore, thread_id: str | None, **fields) -> None:
    await store.record_guard_decision(
        _decision(**fields), session_id=SID_A, thread_id=thread_id, run_id=None, message="hi"
    )


async def test_a_thread_with_only_old_runs_is_stale(store, run_store_clock):
    t0 = run_store_clock.now
    await store.start_run("r1", "old", "default", "fake", SID_A)
    run_store_clock.now = t0 + 40 * DAY
    await store.start_run("r2", "new", "default", "fake", SID_A)

    assert await store.stale_threads(t0 + 30 * DAY) == ["old"]


async def test_a_recent_blocked_follow_up_keeps_an_old_thread(store, run_store_clock):
    t0 = run_store_clock.now
    await store.start_run("r1", "old", "default", "fake", SID_A)
    run_store_clock.now = t0 + 40 * DAY
    await _guard_row(store, "old")

    assert await store.stale_threads(t0 + 30 * DAY) == []


async def test_a_thread_known_only_from_guard_rows_is_found(store, run_store_clock):
    # An allowed message whose run row was never written (start_run logs and goes on).
    t0 = run_store_clock.now
    await _guard_row(store, "guard-only", allowed=True, reason=None)

    assert await store.stale_threads(t0 + 30 * DAY) == ["guard-only"]


async def test_activity_at_the_cutoff_is_kept(store, run_store_clock):
    t0 = run_store_clock.now
    await store.start_run("r1", "edge", "default", "fake", SID_A)

    assert await store.stale_threads(t0) == []
    assert await store.stale_threads(t0 + timedelta(microseconds=1)) == ["edge"]


async def test_stale_threads_come_oldest_first(store, run_store_clock):
    t0 = run_store_clock.now
    await store.start_run("r1", "b", "default", "fake", SID_A)
    run_store_clock.now = t0 + DAY
    await store.start_run("r2", "a", "default", "fake", SID_A)

    assert await store.stale_threads(t0 + 30 * DAY) == ["b", "a"]


async def test_delete_thread_rows_removes_only_that_thread(store, run_store_clock):
    t0 = run_store_clock.now
    await store.start_run("r1", "gone", "default", "fake", SID_A)
    await store.start_run("r2", "gone", "default", "fake", SID_A)
    await _guard_row(store, "gone")
    await store.start_run("r3", "kept", "default", "fake", SID_A)
    await _guard_row(store, "kept")

    deleted = await store.delete_thread_rows("gone", t0 + 30 * DAY)

    assert deleted == RowCounts(runs=2, guard_decisions=1)
    assert await store.owns_thread(SID_A, "gone") is False
    assert await store.owns_thread(SID_A, "kept") is True
    assert await store.stale_threads(t0 + 30 * DAY) == ["kept"]


async def test_delete_thread_rows_keeps_rows_written_after_the_cutoff(store, run_store_clock):
    # A message that lands mid-pass must outlive the pass: without its rows, the
    # checkpoints its turn writes would have nothing that leads a later pass to them.
    t0 = run_store_clock.now
    await store.start_run("r1", "t", "default", "fake", SID_A)
    await _guard_row(store, "t")
    run_store_clock.now = t0 + 40 * DAY
    await store.start_run("r2", "t", "default", "fake", SID_A)

    deleted = await store.delete_thread_rows("t", t0 + 30 * DAY)

    assert deleted == RowCounts(runs=1, guard_decisions=1)
    assert await store.owns_thread(SID_A, "t") is True
    assert await store.stale_threads(t0 + 30 * DAY) == []
    assert await store.stale_threads(t0 + 80 * DAY) == ["t"]


async def test_rows_with_no_thread_go_by_their_own_time(store, run_store_clock):
    t0 = run_store_clock.now
    await _guard_row(store, None)  # a blocked first message
    # A legacy run with no thread: production stopped writing these.
    await store.start_run("r1", None, "default", "fake", SID_A)  # type: ignore[arg-type]
    await store.start_run("r2", "t", "default", "fake", SID_A)
    run_store_clock.now = t0 + 40 * DAY
    await _guard_row(store, None)

    cutoff = t0 + 30 * DAY
    assert await store.count_unthreaded_before(cutoff) == RowCounts(runs=1, guard_decisions=1)
    assert await store.delete_unthreaded_before(cutoff) == RowCounts(runs=1, guard_decisions=1)
    assert await store.count_unthreaded_before(cutoff) == RowCounts(runs=0, guard_decisions=0)
    later = t0 + 50 * DAY
    assert await store.count_unthreaded_before(later) == RowCounts(runs=0, guard_decisions=1)
    assert await store.stale_threads(cutoff) == ["t"]


async def test_the_purge_methods_raise_on_a_broken_store(broken_store, run_store_clock):
    # Unlike the request-path methods: a purge must not report success it didn't have.
    t0 = run_store_clock.now
    with pytest.raises(Exception):
        await broken_store.stale_threads(t0)
    with pytest.raises(Exception):
        await broken_store.delete_thread_rows("mine", t0)
    with pytest.raises(Exception):
        await broken_store.count_unthreaded_before(t0)
    with pytest.raises(Exception):
        await broken_store.delete_unthreaded_before(t0)
    with pytest.raises(Exception):
        await broken_store.unrecorded_threads()


# Each retention method with the arguments that come before its cutoff.
_PURGE_CALLS = {
    "stale_threads": (),
    "delete_thread_rows": ("t",),
    "count_unthreaded_before": (),
    "delete_unthreaded_before": (),
}


@pytest.mark.parametrize("method", list(_PURGE_CALLS))
async def test_the_purge_methods_refuse_a_cutoff_with_no_time_zone(store, method):
    # Compared with stored UTC times, a naive cutoff would shift by the machine's
    # offset. Far in the future, so a store that ignored the zone would delete the row.
    await store.start_run("r1", "t", "default", "fake", SID_A)

    with pytest.raises(ValueError):
        await getattr(store, method)(*_PURGE_CALLS[method], datetime(2100, 1, 1))

    assert await store.owns_thread(SID_A, "t") is True


async def _open_with_saver(
    name: str, stack: AsyncExitStack
) -> tuple[RunStore, BaseCheckpointSaver]:
    """A fresh store and LangGraph's saver on one database, as storage.py opens them."""
    store: RunStore
    saver: BaseCheckpointSaver
    if name == "sqlite":
        conn = await aiosqlite.connect(":memory:")
        stack.push_async_callback(conn.close)
        store, saver = SqliteRunStore(conn), AsyncSqliteSaver(conn)
    else:
        dsn = stack.enter_context(postgres_schema())  # skips without Postgres
        pool = await open_pool(dsn)
        stack.push_async_callback(pool.close)
        store, saver = PostgresRunStore(pool), AsyncPostgresSaver(pool)
    await store.create_table()
    await saver.setup()
    return store, saver


@pytest.fixture(params=_STORES)
async def store_and_saver(request) -> AsyncIterator[tuple[RunStore, BaseCheckpointSaver]]:
    async with AsyncExitStack() as stack:
        yield await _open_with_saver(request.param, stack)


async def _checkpoint(saver: BaseCheckpointSaver, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    await saver.aput(config, empty_checkpoint(), {}, {})


async def test_unrecorded_threads_have_checkpoints_and_no_row(store_and_saver):
    # Both API writes that name a thread are best-effort: a thread whose two writes
    # failed has checkpoints and nothing else.
    store, saver = store_and_saver
    for thread_id in ("unrecorded", "unrecorded", "with-run", "with-guard-row"):
        await _checkpoint(saver, thread_id)
    await store.start_run("r1", "with-run", "default", "fake", SID_A)
    await _guard_row(store, "with-guard-row")
    await store.start_run("r2", "rows-only", "default", "fake", SID_A)

    assert await store.unrecorded_threads() == ["unrecorded"]


async def test_no_checkpoints_table_means_no_unrecorded_threads(store):
    # Memory mode: MemorySaver keeps the checkpoints, outside the store's database.
    await store.start_run("r1", "t", "default", "fake", SID_A)

    assert await store.unrecorded_threads() == []
