from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiosqlite

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


class RunStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create_table(self) -> None:
        await self._conn.execute(_CREATE)
        try:
            await self._conn.execute(_ALTER_ADD_SESSION)
        except aiosqlite.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
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
            started_at = datetime.now(timezone.utc).isoformat()
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
            finished_at = datetime.now(timezone.utc).isoformat()
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
