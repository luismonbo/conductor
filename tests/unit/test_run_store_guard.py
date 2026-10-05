"""Run store support for the input guard: the decision log and the limit queries."""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import aiosqlite
import pytest

import harness.observability.run_store as run_store_module
from harness.core.guard import GuardDecision
from harness.observability.run_store import GUARD_MESSAGE_TEXT_CHARS, RunStore


@pytest.fixture
async def store() -> AsyncGenerator[tuple[RunStore, aiosqlite.Connection], None]:
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    s = RunStore(conn)
    await s.create_table()
    yield s, conn
    await conn.close()


async def _closed_store() -> RunStore:
    conn = await aiosqlite.connect(":memory:")
    s = RunStore(conn)
    await s.create_table()
    await conn.close()
    return s


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


async def test_a_failed_decision_write_returns_none_instead_of_raising():
    s = await _closed_store()
    result = await s.record_guard_decision(
        GuardDecision(allowed=True, stage="classifier"),
        session_id="sid", thread_id=None, run_id=None, message="hi",
    )
    assert result is None


async def test_count_thread_runs(store):
    s, _ = store
    await s.start_run("r1", "t-1", "default", "fake", "sid")
    await s.start_run("r2", "t-1", "default", "fake", "sid")
    await s.start_run("r3", "t-2", "default", "fake", "sid")
    assert await s.count_thread_runs("t-1") == 2
    assert await s.count_thread_runs("t-unknown") == 0


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
    assert await s.tokens_since("2026-10-04T00:00:00+00:00") == 165


async def test_limit_queries_return_zero_when_the_store_fails():
    s = await _closed_store()
    assert await s.count_thread_runs("t-1") == 0
    assert await s.tokens_since("2026-10-04T00:00:00+00:00") == 0


async def test_the_per_message_queries_search_an_index_instead_of_scanning(store):
    """Every chat message runs these; a full scan of runs grows with every run stored."""
    s, conn = store
    for sql, args in [
        (run_store_module._TOKENS_SINCE, ("2026-10-05T00:00:00+00:00",)),
        (run_store_module._COUNT_THREAD_RUNS, ("t-1",)),
        (run_store_module._OWNS_THREAD, ("t-1", "sid")),
    ]:
        async with conn.execute("EXPLAIN QUERY PLAN " + sql, args) as cur:
            plan = " | ".join(row[-1] for row in await cur.fetchall())
        assert plan.startswith("SEARCH runs USING") and "INDEX" in plan, plan
