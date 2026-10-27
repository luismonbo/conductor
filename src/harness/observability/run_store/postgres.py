"""PostgresRunStore: the run store in Postgres, on the pool the checkpointer shares.

The rows SqliteRunStore keeps, in Postgres types: TIMESTAMPTZ times, a BOOLEAN
verdict, JSONB scores. Times read back in UTC whatever the session's TimeZone.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from harness.adapters.postgres.pool import PostgresPool
from harness.core.guard import GuardDecision
from harness.observability.run_store import storable_text
from harness.observability.token_accumulator import TokenAccumulator

logger = logging.getLogger(__name__)

# Run on every start, so each statement is idempotent. A later column joins this
# list as "ALTER TABLE ... ADD COLUMN IF NOT EXISTS": two tables need no migration
# framework. session_id is in runs from the start, so SQLite's ALTER has no twin.
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id         TEXT PRIMARY KEY,
        thread_id      TEXT,
        agent          TEXT,
        backend        TEXT,
        started_at     TIMESTAMPTZ NOT NULL,
        finished_at    TIMESTAMPTZ,
        stopped_reason TEXT,
        input_tokens   INTEGER,
        output_tokens  INTEGER,
        iterations     INTEGER,
        session_id     TEXT
    )
    """,
    # Every chat message runs up to three lookups on runs (daily budget, thread
    # turns, ownership); these keep each one an index search.
    "CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs (started_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_thread_session ON runs (thread_id, session_id)",
    """
    CREATE TABLE IF NOT EXISTS guard_decisions (
        decision_id     TEXT PRIMARY KEY,
        created_at      TIMESTAMPTZ NOT NULL,
        session_id      TEXT,
        thread_id       TEXT,
        run_id          TEXT,
        allowed         BOOLEAN NOT NULL,
        reason          TEXT,
        stage           TEXT NOT NULL,
        probabilities   JSONB,
        battery_version TEXT,
        model           TEXT,
        request_id      TEXT,
        latency_ms      INTEGER,
        input_tokens    INTEGER,
        error           TEXT,
        message_chars   INTEGER NOT NULL,
        message_text    TEXT NOT NULL
    )
    """,
)

_INSERT_RUN = """
INSERT INTO runs (run_id, thread_id, agent, backend, started_at, session_id)
VALUES (%s, %s, %s, %s, %s, %s)
"""

_FINISH_RUN = """
UPDATE runs
SET finished_at = %s, stopped_reason = %s,
    input_tokens = %s, output_tokens = %s, iterations = %s
WHERE run_id = %s
"""

_LIST_THREADS = """
SELECT thread_id, MAX(started_at) AS last_at, COUNT(*) AS runs
FROM runs
WHERE thread_id IS NOT NULL AND session_id = %s
GROUP BY thread_id
ORDER BY last_at DESC
LIMIT %s
"""

_OWNS_THREAD = (
    "SELECT 1 AS owned FROM runs WHERE thread_id = %s AND session_id = %s LIMIT 1"
)

_COUNT_THREAD_RUNS = "SELECT COUNT(*) AS runs FROM runs WHERE thread_id = %s"

# Runs still in flight have NULL token counts and count as 0 until they finish.
_TOKENS_SINCE = """
SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0) AS tokens
FROM runs WHERE started_at >= %s
"""

_INSERT_GUARD_DECISION = """
INSERT INTO guard_decisions (
    decision_id, created_at, session_id, thread_id, run_id, allowed, reason, stage,
    probabilities, battery_version, model, request_id, latency_ms, input_tokens,
    error, message_chars, message_text
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(moment: datetime) -> str:
    """A timestamp as the UTC ISO-8601 text SqliteRunStore stores and the UI parses."""
    return moment.astimezone(timezone.utc).isoformat()


class PostgresRunStore:
    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def create_table(self) -> None:
        async with self._pool.connection() as conn:
            for statement in _SCHEMA:
                await conn.execute(statement)

    async def start_run(
        self, run_id: str, thread_id: str, agent: str, backend: str, session_id: str
    ) -> None:
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    _INSERT_RUN, (run_id, thread_id, agent, backend, _now(), session_id)
                )
        except Exception:
            logger.exception("RunStore.start_run failed for run_id=%s", run_id)

    async def list_threads(self, session_id: str, limit: int = 50) -> list[dict]:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(_LIST_THREADS, (session_id, limit))
                rows = await cur.fetchall()
        except Exception:
            logger.exception("RunStore.list_threads failed")
            return []
        return [
            {
                "thread_id": row["thread_id"],
                "last_at": _iso_utc(row["last_at"]),
                "runs": row["runs"],
            }
            for row in rows
        ]

    async def owns_thread(self, session_id: str, thread_id: str) -> bool:
        """True only when this session started at least one run on this thread.

        Returns False on any error: this gates access, so it must fail closed.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(_OWNS_THREAD, (thread_id, session_id))
                return await cur.fetchone() is not None
        except Exception:
            logger.exception("RunStore.owns_thread failed for thread_id=%s", thread_id)
            return False

    async def finish_run(
        self, run_id: str, accumulator: TokenAccumulator, stopped_reason: str
    ) -> None:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    _FINISH_RUN,
                    (
                        _now(),
                        stopped_reason,
                        accumulator.input_tokens,
                        accumulator.output_tokens,
                        accumulator.iterations,
                        run_id,
                    ),
                )
                if cur.rowcount == 0:
                    logger.warning("finish_run: no row updated for run_id=%s", run_id)
        except Exception:
            logger.exception("RunStore.finish_run failed for run_id=%s", run_id)

    async def record_guard_decision(
        self,
        decision: GuardDecision,
        *,
        session_id: str,
        thread_id: str | None,
        run_id: str | None,
        message: str,
    ) -> str | None:
        """Persist one input-guard decision; return its id, or None on failure.

        Never raises: a failed write is logged and must not break the chat.
        """
        decision_id = str(uuid.uuid4())
        probabilities = (
            Jsonb(dict(decision.probabilities))
            if decision.probabilities is not None
            else None
        )
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    _INSERT_GUARD_DECISION,
                    (
                        decision_id,
                        _now(),
                        session_id,
                        thread_id,
                        run_id,
                        decision.allowed,
                        decision.reason,
                        decision.stage,
                        probabilities,
                        decision.battery_version,
                        decision.model,
                        decision.request_id,
                        decision.latency_ms,
                        decision.input_tokens,
                        decision.error,
                        len(message),
                        storable_text(message),
                    ),
                )
            return decision_id
        except Exception:
            logger.exception("RunStore.record_guard_decision failed")
            return None

    async def count_thread_runs(self, thread_id: str) -> int:
        """Runs recorded on this thread. 0 on error, so the turn limit fails open."""
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(_COUNT_THREAD_RUNS, (thread_id,))
                row = await cur.fetchone()
            return int(row["runs"]) if row else 0
        except Exception:
            logger.exception("RunStore.count_thread_runs failed for thread_id=%s", thread_id)
            return 0

    async def tokens_since(self, since: datetime) -> int:
        """Input + output tokens of runs started at or after `since`.

        Compared as timestamps, so the session's TimeZone doesn't matter. 0 on
        error, so the daily budget fails open.
        """
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(_TOKENS_SINCE, (since,))
                row = await cur.fetchone()
            return int(row["tokens"]) if row else 0
        except Exception:
            logger.exception("RunStore.tokens_since failed")
            return 0
