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

from harness.core.guard import GuardDecision
from harness.observability.run_store import RunStore
from harness.observability.run_store.sqlite import SqliteRunStore
from harness.observability.token_accumulator import TokenAccumulator

SID_A = "session-a"
SID_B = "session-b"

_STORES = ["sqlite"]


async def _open(name: str, stack: AsyncExitStack) -> RunStore:
    """A fresh store with its tables; `stack` closes it after the test."""
    conn = await aiosqlite.connect(":memory:")
    stack.push_async_callback(conn.close)
    opened = SqliteRunStore(conn)
    await opened.create_table()
    return opened


@pytest.fixture(params=_STORES)
async def store(request) -> AsyncIterator[RunStore]:
    async with AsyncExitStack() as stack:
        yield await _open(request.param, stack)


@pytest.fixture(params=_STORES)
async def broken_store(request) -> AsyncIterator[RunStore]:
    """A store whose every query fails: its connection, or its whole pool, is closed."""
    async with AsyncExitStack() as stack:
        broken = await _open(request.param, stack)
        await broken.start_run("r1", "mine", "default", "fake", SID_A)
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


async def test_count_thread_runs(store):
    await store.start_run("r1", "t-1", "default", "fake", SID_A)
    await store.start_run("r2", "t-1", "default", "fake", SID_A)
    await store.start_run("r3", "t-2", "default", "fake", SID_A)
    assert await store.count_thread_runs("t-1") == 2
    assert await store.count_thread_runs("t-unknown") == 0


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


async def test_a_failing_store_fails_closed_on_ownership_and_open_on_limits(broken_store):
    assert await broken_store.owns_thread(SID_A, "mine") is False
    assert await broken_store.count_thread_runs("mine") == 0
    day_ago = datetime.now(timezone.utc) - timedelta(days=1)
    assert await broken_store.tokens_since(day_ago) == 0
    assert await broken_store.list_threads(SID_A) == []


async def test_a_failing_store_never_raises_on_writes(broken_store):
    await broken_store.start_run("r2", "t2", "default", "fake", SID_A)
    await broken_store.finish_run("r2", _tokens(1, 1), "final_answer")
    assert await broken_store.record_guard_decision(
        _decision(), session_id=SID_A, thread_id=None, run_id=None, message="hi"
    ) is None
