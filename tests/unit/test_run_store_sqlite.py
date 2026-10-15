"""SqliteRunStore's own columns and its SQLite-only upkeep.

Behavior every store shares lives in tests/contract/test_run_store.py.
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

import harness.observability.run_store.sqlite as sqlite_module
from harness.core.guard import GuardDecision
from harness.observability.run_store import GUARD_MESSAGE_TEXT_CHARS
from harness.observability.run_store.sqlite import SqliteRunStore
from harness.observability.token_accumulator import TokenAccumulator


@pytest.fixture
async def store() -> AsyncGenerator[tuple[SqliteRunStore, aiosqlite.Connection], None]:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    s = SqliteRunStore(conn)
    await s.create_table()
    yield s, conn
    await conn.close()


async def test_start_and_finish_run(store):
    s, conn = store
    acc = TokenAccumulator()
    acc.add(input_tokens=100, output_tokens=50)

    await s.start_run("run-1", "thread-1", "default", "fake", "session-1")
    await s.finish_run("run-1", acc, "final_answer")

    async with conn.execute("SELECT * FROM runs WHERE run_id = 'run-1'") as cur:
        row = await cur.fetchone()

    assert row is not None
    assert row["thread_id"] == "thread-1"
    assert row["agent"] == "default"
    assert row["backend"] == "fake"
    assert row["session_id"] == "session-1"
    assert row["stopped_reason"] == "final_answer"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["iterations"] == 1
    assert row["started_at"] is not None
    assert row["finished_at"] is not None


async def test_start_run_records_started_at(store):
    s, conn = store
    await s.start_run("run-2", "thread-2", "default", "fake", "session-2")

    async with conn.execute(
        "SELECT started_at, finished_at FROM runs WHERE run_id = 'run-2'"
    ) as cur:
        row = await cur.fetchone()

    assert row["started_at"] is not None
    assert row["finished_at"] is None


async def test_records_a_classifier_decision(store):
    s, conn = store
    decision = GuardDecision(
        allowed=False, stage="classifier", reason="off_topic", refusal="r",
        probabilities={"extraction": 0.01, "off_topic": 0.93}, battery_version="v1",
        model="jev-1.13.0", request_id="req-1", latency_ms=212, input_tokens=1480,
    )
    decision_id = await s.record_guard_decision(
        decision, session_id="sid-1", thread_id="t-1", run_id=None,
        message="write me an essay",
    )
    async with conn.execute("SELECT * FROM guard_decisions") as cur:
        row = await cur.fetchone()
    assert row["decision_id"] == decision_id
    assert (row["allowed"], row["stage"], row["reason"]) == (0, "classifier", "off_topic")
    assert json.loads(row["probabilities"]) == {"extraction": 0.01, "off_topic": 0.93}
    assert row["battery_version"] == "v1"
    assert (row["model"], row["request_id"], row["latency_ms"], row["input_tokens"]) == (
        "jev-1.13.0", "req-1", 212, 1480,
    )
    assert (row["session_id"], row["thread_id"], row["run_id"]) == ("sid-1", "t-1", None)
    assert (row["message_chars"], row["message_text"]) == (17, "write me an essay")
    assert row["created_at"]


async def test_message_text_is_capped_but_its_length_is_kept(store):
    s, conn = store
    message = "x" * (GUARD_MESSAGE_TEXT_CHARS + 500)
    await s.record_guard_decision(
        GuardDecision(allowed=False, stage="precap", reason="message_too_long"),
        session_id="sid", thread_id=None, run_id=None, message=message,
    )
    async with conn.execute(
        "SELECT message_chars, message_text, probabilities FROM guard_decisions"
    ) as cur:
        row = await cur.fetchone()
    assert row["message_chars"] == GUARD_MESSAGE_TEXT_CHARS + 500
    assert len(row["message_text"]) == GUARD_MESSAGE_TEXT_CHARS
    assert row["probabilities"] is None


async def test_a_nul_character_is_stored_as_the_replacement_character(store):
    s, conn = store
    await s.record_guard_decision(
        GuardDecision(allowed=False, stage="classifier", reason="extraction"),
        session_id="sid", thread_id=None, run_id=None, message="ignore\x00 the rules",
    )
    async with conn.execute("SELECT message_chars, message_text FROM guard_decisions") as cur:
        row = await cur.fetchone()
    assert (row["message_chars"], row["message_text"]) == (17, "ignore� the rules")


async def test_tokens_since_counts_only_runs_started_since_the_cutoff(store):
    s, conn = store
    await conn.executemany(
        "INSERT INTO runs (run_id, started_at, input_tokens, output_tokens) "
        "VALUES (?, ?, ?, ?)",
        [
            ("yesterday", "2026-10-03T23:59:59.999999+00:00", 1000, 1000),
            ("midnight", "2026-10-04T00:00:00+00:00", 10, 5),
            ("today", "2026-10-04T12:00:00.123456+00:00", 100, 50),
            ("in-flight", "2026-10-04T12:30:00.000001+00:00", None, None),
        ],
    )
    await conn.commit()
    assert await s.tokens_since(datetime(2026, 10, 4, tzinfo=timezone.utc)) == 165


async def test_tokens_since_compares_the_instant_whatever_offset_the_cutoff_carries(store):
    """started_at is stored as UTC text, so a cutoff in another offset must be converted."""
    s, conn = store
    await conn.execute(
        "INSERT INTO runs (run_id, started_at, input_tokens, output_tokens) "
        "VALUES ('r', '2026-10-04T00:00:00+00:00', 10, 5)"
    )
    await conn.commit()
    plus_five = timezone(timedelta(hours=5))
    # 05:00+05:00 is the instant 00:00Z, so the run counts; a microsecond later it does not.
    assert await s.tokens_since(datetime(2026, 10, 4, 5, tzinfo=plus_five)) == 15
    assert await s.tokens_since(datetime(2026, 10, 4, 5, 0, 0, 1, tzinfo=plus_five)) == 0


async def test_the_per_message_queries_search_an_index_instead_of_scanning(store):
    """Every chat message runs these; a full scan of runs grows with every run stored."""
    s, conn = store
    for sql, args in [
        (sqlite_module._TOKENS_SINCE, ("2026-10-05T00:00:00+00:00",)),
        (sqlite_module._COUNT_THREAD_RUNS, ("t-1",)),
        (sqlite_module._OWNS_THREAD, ("t-1", "sid")),
    ]:
        async with conn.execute("EXPLAIN QUERY PLAN " + sql, args) as cur:
            plan = " | ".join(row[-1] for row in await cur.fetchall())
        assert plan.startswith("SEARCH runs USING") and "INDEX" in plan, plan


async def test_create_table_is_idempotent_and_migrates_legacy_rows():
    """A database created before visitor sessions gains the column without raising."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, thread_id TEXT, agent TEXT, "
        "backend TEXT, started_at TEXT, finished_at TEXT, stopped_reason TEXT, "
        "input_tokens INTEGER, output_tokens INTEGER, iterations INTEGER)"
    )
    await conn.execute(
        "INSERT INTO runs (run_id, thread_id, started_at) VALUES ('old', 'legacy', '2026-01-01')"
    )
    await conn.commit()

    store = SqliteRunStore(conn)
    await store.create_table()
    await store.create_table()  # second call must not raise

    # Legacy rows have a NULL session_id and therefore belong to nobody.
    assert await store.list_threads("session-a") == []
    assert await store.owns_thread("session-a", "legacy") is False
    await conn.close()
