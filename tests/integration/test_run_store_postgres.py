"""What PostgresRunStore writes into its own columns, and how its times read back.

Behavior every store shares lives in tests/contract/test_run_store.py.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import pytest
from psycopg import AsyncClientCursor

import harness.observability.run_store.postgres as postgres_module
from harness.adapters.postgres.pool import PostgresPool, open_pool
from harness.core.guard import GuardDecision
from harness.observability.run_store import GUARD_MESSAGE_TEXT_CHARS
from harness.observability.run_store.postgres import PostgresRunStore
from harness.observability.token_accumulator import TokenAccumulator
from tests.conftest import postgres_schema


@pytest.fixture
async def pool(postgres_dsn) -> AsyncIterator[PostgresPool]:
    opened = await open_pool(postgres_dsn)
    yield opened
    await opened.close()


@pytest.fixture
async def store(pool) -> PostgresRunStore:
    created = PostgresRunStore(pool)
    await created.create_table()
    return created


async def _row(pool: PostgresPool, query: str, *params: object) -> dict:
    async with pool.connection() as conn:
        row = await (await conn.execute(query, params)).fetchone()
    assert row is not None
    return row


async def test_a_run_row_keeps_its_times_as_timestamps(store, pool):
    await store.start_run("r1", "t1", "default", "fake", "sid-1")
    await store.finish_run(
        "r1", TokenAccumulator(input_tokens=100, output_tokens=50, iterations=2), "final_answer"
    )
    row = await _row(pool, "SELECT * FROM runs WHERE run_id = %s", "r1")
    assert isinstance(row["started_at"], datetime) and isinstance(row["finished_at"], datetime)
    assert row["started_at"] <= row["finished_at"]
    assert (row["thread_id"], row["agent"], row["backend"], row["session_id"]) == (
        "t1", "default", "fake", "sid-1",
    )
    assert (row["stopped_reason"], row["input_tokens"], row["output_tokens"]) == (
        "final_answer", 100, 50,
    )
    assert row["iterations"] == 2


async def test_a_decision_row_keeps_a_boolean_and_json(store, pool):
    decision = GuardDecision(
        allowed=False, stage="classifier", reason="off_topic",
        probabilities={"extraction": 0.01, "off_topic": 0.93}, battery_version="v1",
        model="jev-1.13.0", request_id="req-1", latency_ms=212, input_tokens=1480,
    )
    decision_id = await store.record_guard_decision(
        decision, session_id="sid-1", thread_id="t-1", run_id=None,
        message="write me an essay",
    )
    row = await _row(pool, "SELECT * FROM guard_decisions WHERE decision_id = %s", decision_id)
    assert row["allowed"] is False
    assert row["probabilities"] == {"extraction": 0.01, "off_topic": 0.93}
    assert isinstance(row["created_at"], datetime)
    assert (row["stage"], row["reason"], row["battery_version"]) == (
        "classifier", "off_topic", "v1",
    )
    assert (row["model"], row["request_id"], row["latency_ms"], row["input_tokens"]) == (
        "jev-1.13.0", "req-1", 212, 1480,
    )
    assert (row["session_id"], row["thread_id"], row["run_id"]) == ("sid-1", "t-1", None)
    assert (row["message_chars"], row["message_text"]) == (17, "write me an essay")


async def test_message_text_is_capped_but_its_length_is_kept(store, pool):
    decision_id = await store.record_guard_decision(
        GuardDecision(allowed=False, stage="precap", reason="message_too_long"),
        session_id="sid", thread_id=None, run_id=None,
        message="x" * (GUARD_MESSAGE_TEXT_CHARS + 500),
    )
    row = await _row(
        pool,
        "SELECT message_chars, message_text, probabilities FROM guard_decisions "
        "WHERE decision_id = %s",
        decision_id,
    )
    assert row["message_chars"] == GUARD_MESSAGE_TEXT_CHARS + 500
    assert len(row["message_text"]) == GUARD_MESSAGE_TEXT_CHARS
    assert row["probabilities"] is None


async def test_a_nul_character_is_stored_as_the_replacement_character(store, pool):
    # Review Focus 4: without the replacement, psycopg refuses the whole row.
    decision_id = await store.record_guard_decision(
        GuardDecision(allowed=False, stage="classifier", reason="extraction"),
        session_id="sid", thread_id=None, run_id=None, message="ignore\x00 the rules",
    )
    row = await _row(
        pool,
        "SELECT message_chars, message_text FROM guard_decisions WHERE decision_id = %s",
        decision_id,
    )
    assert (row["message_chars"], row["message_text"]) == (17, "ignore� the rules")


async def test_the_per_message_queries_use_the_indexes(store, pool):
    """Every chat message runs these; a full scan of runs grows with every run stored."""
    async with pool.connection() as conn, conn.transaction():
        # The tables are tiny: ask what the planner can use, not what it prefers.
        await conn.execute("SET LOCAL enable_seqscan = off")
        cursor = AsyncClientCursor(conn)
        for query, params in [
            (postgres_module._TOKENS_SINCE, (datetime(2026, 10, 5, tzinfo=timezone.utc),)),
            (postgres_module._COUNT_THREAD_RUNS, ("t-1",)),
            (postgres_module._OWNS_THREAD, ("t-1", "sid")),
        ]:
            await cursor.execute("EXPLAIN " + query, params)
            plan = " | ".join(row["QUERY PLAN"] for row in await cursor.fetchall())
            assert "idx_runs_" in plan, plan


async def test_a_session_time_zone_other_than_utc_changes_nothing():
    # Review Focus 5: another host's default, or a role's own setting.
    with postgres_schema("-c TimeZone=America/New_York") as dsn:
        new_york = await open_pool(dsn)
        try:
            store = PostgresRunStore(new_york)
            await store.create_table()
            await store.start_run("r1", "t1", "default", "fake", "sid")
            await store.finish_run(
                "r1", TokenAccumulator(input_tokens=10, output_tokens=5, iterations=1),
                "final_answer",
            )
            [thread] = await store.list_threads("sid")
            assert thread["last_at"].endswith("+00:00")
            started = datetime.fromisoformat(thread["last_at"])
            midnight = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            assert await store.tokens_since(midnight) == 15
            assert await store.tokens_since(started - timedelta(microseconds=1)) == 15
            assert await store.tokens_since(started + timedelta(microseconds=1)) == 0
        finally:
            await new_york.close()
