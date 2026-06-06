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

_INSERT = """
INSERT INTO runs (run_id, thread_id, agent, backend, started_at)
VALUES (?, ?, ?, ?, ?)
"""

_UPDATE = """
UPDATE runs
SET finished_at = ?, stopped_reason = ?,
    input_tokens = ?, output_tokens = ?, iterations = ?
WHERE run_id = ?
"""


class RunStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create_table(self) -> None:
        await self._conn.execute(_CREATE)
        await self._conn.commit()

    async def start_run(
        self,
        run_id: str,
        thread_id: str,
        agent: str,
        backend: str,
    ) -> None:
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            await self._conn.execute(_INSERT, (run_id, thread_id, agent, backend, started_at))
            await self._conn.commit()
        except Exception:
            logger.exception("RunStore.start_run failed for run_id=%s", run_id)

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
