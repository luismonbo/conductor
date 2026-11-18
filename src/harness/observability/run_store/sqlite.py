"""SqliteRunStore: the run store in harness.sqlite (local runs) or in memory (tests)."""
from __future__ import annotations

import contextlib
import json
import logging
import uuid
from datetime import datetime, timezone

import aiosqlite

from harness.core.guard import GuardDecision
from harness.observability.run_store import RowCounts, storable_text
from harness.observability.token_accumulator import TokenAccumulator

logger = logging.getLogger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    thread_id     TEXT,
    agent         TEXT,
    backend       TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    stopped_reason TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    iterations    INTEGER
)
"""

# CREATE TABLE IF NOT EXISTS will not add a column to a table that already
# exists, so an existing database needs an explicit ALTER. Re-running it raises
# "duplicate column name", which is the success case on every run after the
# first. Legacy rows get NULL and therefore belong to no session — which is the
# desired outcome: pre-existing local development threads become invisible.
_ALTER_ADD_SESSION = "ALTER TABLE runs ADD COLUMN session_id TEXT"

_INSERT = """
INSERT INTO runs (run_id, thread_id, agent, backend, started_at, session_id)
VALUES (?, ?, ?, ?, ?, ?)
"""

_UPDATE = """
UPDATE runs
SET finished_at = ?, stopped_reason = ?,
    input_tokens = ?, output_tokens = ?, iterations = ?
WHERE run_id = ?
"""

_LIST_THREADS = """
SELECT thread_id, MAX(started_at) AS last_at, COUNT(*) AS runs
FROM runs
WHERE thread_id IS NOT NULL AND session_id = ?
GROUP BY thread_id
ORDER BY last_at DESC
LIMIT ?
"""

_OWNS_THREAD = "SELECT 1 FROM runs WHERE thread_id = ? AND session_id = ? LIMIT 1"

# Every chat message runs up to three lookups on runs (daily budget, thread
# turns, ownership); without these each is a full table scan. IF NOT EXISTS
# makes them idempotent, so existing databases gain them on the next start.
_CREATE_RUN_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_thread_session ON runs(thread_id, session_id)",
)

# One row per screened chat message (input guard, SL-2). Created with IF NOT
# EXISTS, so existing databases need no migration.
_CREATE_GUARD_DECISIONS = """
CREATE TABLE IF NOT EXISTS guard_decisions (
    decision_id     TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    session_id      TEXT,
    thread_id       TEXT,
    run_id          TEXT,
    allowed         INTEGER NOT NULL,
    reason          TEXT,
    stage           TEXT NOT NULL,
    probabilities   TEXT,
    battery_version TEXT,
    model           TEXT,
    request_id      TEXT,
    latency_ms      INTEGER,
    input_tokens    INTEGER,
    error           TEXT,
    message_chars   INTEGER NOT NULL,
    message_text    TEXT NOT NULL
)
"""

_INSERT_GUARD_DECISION = """
INSERT INTO guard_decisions (
    decision_id, created_at, session_id, thread_id, run_id, allowed, reason, stage,
    probabilities, battery_version, model, request_id, latency_ms, input_tokens,
    error, message_chars, message_text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Runs still in flight have NULL token counts and count as 0 until they finish.
_TOKENS_SINCE = """
SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0)
FROM runs WHERE started_at >= ?
"""

# Retention (RunStore's protocol says what each returns). A thread's latest activity
# is the newest of its runs and guard_decisions rows. Times compare as text, as
# _TOKENS_SINCE does: every one is written by _now().isoformat().
_STALE_THREADS = """
SELECT thread_id
FROM (
    SELECT thread_id, started_at AS at FROM runs WHERE thread_id IS NOT NULL
    UNION ALL
    SELECT thread_id, created_at AS at FROM guard_decisions WHERE thread_id IS NOT NULL
) AS activity
GROUP BY thread_id
HAVING MAX(at) < ?
ORDER BY MAX(at), thread_id
"""

_DELETE_THREAD_RUNS = "DELETE FROM runs WHERE thread_id = ?"
_DELETE_THREAD_GUARD_DECISIONS = "DELETE FROM guard_decisions WHERE thread_id = ?"

_COUNT_UNTHREADED_RUNS = "SELECT COUNT(*) FROM runs WHERE thread_id IS NULL AND started_at < ?"
_COUNT_UNTHREADED_GUARD_DECISIONS = (
    "SELECT COUNT(*) FROM guard_decisions WHERE thread_id IS NULL AND created_at < ?"
)
_DELETE_UNTHREADED_RUNS = "DELETE FROM runs WHERE thread_id IS NULL AND started_at < ?"
_DELETE_UNTHREADED_GUARD_DECISIONS = (
    "DELETE FROM guard_decisions WHERE thread_id IS NULL AND created_at < ?"
)


def _now() -> datetime:
    """The time every row is stamped with (tests replace it: run_store_clock)."""
    return datetime.now(timezone.utc)


def _iso_utc(moment: datetime) -> str:
    """`moment` in the format started_at is stored in, so the two compare as text."""
    return moment.astimezone(timezone.utc).isoformat()


class SqliteRunStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create_table(self) -> None:
        await self._conn.execute(_CREATE)
        try:
            await self._conn.execute(_ALTER_ADD_SESSION)
        except aiosqlite.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
        for create_index in _CREATE_RUN_INDEXES:  # after the ALTER: one indexes session_id
            await self._conn.execute(create_index)
        await self._conn.execute(_CREATE_GUARD_DECISIONS)
        await self._conn.commit()

    async def start_run(
        self,
        run_id: str,
        thread_id: str,
        agent: str,
        backend: str,
        session_id: str,
    ) -> None:
        try:
            started_at = _now().isoformat()
            await self._conn.execute(
                _INSERT, (run_id, thread_id, agent, backend, started_at, session_id)
            )
            await self._conn.commit()
        except Exception:
            logger.exception("RunStore.start_run failed for run_id=%s", run_id)

    async def list_threads(self, session_id: str, limit: int = 50) -> list[dict]:
        try:
            cur = await self._conn.execute(_LIST_THREADS, (session_id, limit))
            rows = await cur.fetchall()
            return [
                {"thread_id": r[0], "last_at": r[1], "runs": r[2]} for r in rows
            ]
        except Exception:
            logger.exception("RunStore.list_threads failed")
            return []

    async def owns_thread(self, session_id: str, thread_id: str) -> bool:
        """True only when this session started at least one run on this thread.

        Returns False on any error: this gates access, so it must fail closed.
        """
        try:
            cur = await self._conn.execute(_OWNS_THREAD, (thread_id, session_id))
            return await cur.fetchone() is not None
        except Exception:
            logger.exception("RunStore.owns_thread failed for thread_id=%s", thread_id)
            return False

    async def finish_run(
        self,
        run_id: str,
        accumulator: TokenAccumulator,
        stopped_reason: str,
    ) -> None:
        try:
            finished_at = _now().isoformat()
            cur = await self._conn.execute(
                _UPDATE,
                (
                    finished_at,
                    stopped_reason,
                    accumulator.input_tokens,
                    accumulator.output_tokens,
                    accumulator.iterations,
                    run_id,
                ),
            )
            await self._conn.commit()
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
            json.dumps(dict(decision.probabilities))
            if decision.probabilities is not None
            else None
        )
        try:
            await self._conn.execute(
                _INSERT_GUARD_DECISION,
                (
                    decision_id,
                    _now().isoformat(),
                    session_id,
                    thread_id,
                    run_id,
                    int(decision.allowed),
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
            await self._conn.commit()
            return decision_id
        except Exception:
            logger.exception("RunStore.record_guard_decision failed")
            return None

    async def tokens_since(self, since: datetime) -> int:
        """Input + output tokens of runs started at or after `since`.

        Every started_at is written by datetime.now(timezone.utc).isoformat(), so
        `since` in the same format compares correctly as text. 0 on error, so the
        daily budget fails open.
        """
        try:
            cur = await self._conn.execute(_TOKENS_SINCE, (_iso_utc(since),))
            row = await cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            logger.exception("RunStore.tokens_since failed")
            return 0

    async def stale_threads(self, cutoff: datetime) -> list[str]:
        cur = await self._conn.execute(_STALE_THREADS, (_iso_utc(cutoff),))
        return [row[0] for row in await cur.fetchall()]

    async def delete_thread_rows(self, thread_id: str) -> RowCounts:
        return await self._delete_both(
            (_DELETE_THREAD_RUNS, _DELETE_THREAD_GUARD_DECISIONS), (thread_id,)
        )

    async def count_unthreaded_before(self, cutoff: datetime) -> RowCounts:
        at = (_iso_utc(cutoff),)
        runs = await (await self._conn.execute(_COUNT_UNTHREADED_RUNS, at)).fetchone()
        decisions = await (
            await self._conn.execute(_COUNT_UNTHREADED_GUARD_DECISIONS, at)
        ).fetchone()
        return RowCounts(runs=int(runs[0]), guard_decisions=int(decisions[0]))

    async def delete_unthreaded_before(self, cutoff: datetime) -> RowCounts:
        return await self._delete_both(
            (_DELETE_UNTHREADED_RUNS, _DELETE_UNTHREADED_GUARD_DECISIONS), (_iso_utc(cutoff),)
        )

    async def _delete_both(self, statements: tuple[str, str], params: tuple) -> RowCounts:
        """Run a runs DELETE and a guard_decisions DELETE as one transaction.

        sqlite3 opens the transaction on the first DELETE; commit ends it, and a
        failure rolls both back before the error propagates.
        """
        runs_sql, decisions_sql = statements
        try:
            runs = await self._conn.execute(runs_sql, params)
            decisions = await self._conn.execute(decisions_sql, params)
            await self._conn.commit()
        except Exception:
            # A closed connection can't roll back either; the original error matters.
            with contextlib.suppress(Exception):
                await self._conn.rollback()
            raise
        return RowCounts(runs=runs.rowcount, guard_decisions=decisions.rowcount)
