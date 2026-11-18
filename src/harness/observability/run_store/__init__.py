"""The run store: one row per agent run, and one per screened chat message.

`RunStore` is the protocol the API uses. Two stores implement it: SqliteRunStore
(run_store/sqlite.py: local runs and tests) and PostgresRunStore
(run_store/postgres.py: production). tests/contract/test_run_store.py holds both
to the same behavior, failures included:
- owns_thread returns False on any error: it gates access, so it fails closed.
- tokens_since returns 0, so the daily budget fails open.
- list_threads returns []; start_run, finish_run and record_guard_decision log
  and never raise (record_guard_decision returns None).
- The retention methods (stale_threads, delete_thread_rows, count_unthreaded_before,
  delete_unthreaded_before) raise: the purge (orchestration/retention.py) must not
  report a pass that deleted nothing as a success.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from harness.core.guard import GuardDecision
from harness.observability.token_accumulator import TokenAccumulator

# Decision rows keep this many characters of the screened message.
GUARD_MESSAGE_TEXT_CHARS = 2000


def storable_text(message: str) -> str:
    """The part of `message` a decision row keeps, with NUL characters replaced.

    Postgres text columns reject NUL, and a probing message that carries one must
    still leave its row. Both stores store the same text.
    """
    return message[:GUARD_MESSAGE_TEXT_CHARS].replace("\x00", "�")


@dataclass(frozen=True)
class RowCounts:
    """How many rows of each table a retention query counted or deleted."""

    runs: int
    guard_decisions: int


class RunStore(Protocol):
    async def create_table(self) -> None:
        """Create the tables and indexes if missing. Idempotent; raises on failure."""
        ...

    async def start_run(
        self, run_id: str, thread_id: str, agent: str, backend: str, session_id: str
    ) -> None: ...

    async def finish_run(
        self, run_id: str, accumulator: TokenAccumulator, stopped_reason: str
    ) -> None: ...

    async def list_threads(self, session_id: str, limit: int = 50) -> list[dict]:
        """This session's threads, newest first: thread_id, last_at (UTC ISO-8601), runs."""
        ...

    async def owns_thread(self, session_id: str, thread_id: str) -> bool:
        """True only when this session started at least one run on this thread."""
        ...

    async def record_guard_decision(
        self,
        decision: GuardDecision,
        *,
        session_id: str,
        thread_id: str | None,
        run_id: str | None,
        message: str,
    ) -> str | None:
        """Persist one input-guard decision; return its id, or None on failure."""
        ...

    async def tokens_since(self, since: datetime) -> int:
        """Input + output tokens of runs started at or after `since` (timezone-aware)."""
        ...

    # Retention (docs/superpowers/specs/2026-10-14-retention-purge-design.md). A
    # thread's latest activity is the newest of its runs.started_at and
    # guard_decisions.created_at: a blocked follow-up still means the visitor was
    # there. `cutoff` is timezone-aware. All four raise on failure.

    async def stale_threads(self, cutoff: datetime) -> list[str]:
        """Ids of the threads whose latest activity is before `cutoff`, oldest first."""
        ...

    async def delete_thread_rows(self, thread_id: str) -> RowCounts:
        """Delete the thread's runs and guard_decisions rows in one transaction."""
        ...

    async def count_unthreaded_before(self, cutoff: datetime) -> RowCounts:
        """Count the rows with no thread (blocked first messages, legacy runs) before `cutoff`."""
        ...

    async def delete_unthreaded_before(self, cutoff: datetime) -> RowCounts:
        """Delete the rows count_unthreaded_before counts; return how many went."""
        ...
