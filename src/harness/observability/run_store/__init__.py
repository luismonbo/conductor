"""The run store: one row per agent run, and one per screened chat message.

`RunStore` is the protocol the API uses. Two stores implement it: SqliteRunStore
(run_store/sqlite.py: local runs and tests) and PostgresRunStore
(run_store/postgres.py: production). tests/contract/test_run_store.py holds both
to the same behavior, failures included:
- owns_thread returns False on any error: it gates access, so it fails closed.
- count_thread_runs and tokens_since return 0, so the limits fail open.
- list_threads returns []; start_run, finish_run and record_guard_decision log
  and never raise (record_guard_decision returns None).
"""
from __future__ import annotations

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

    async def count_thread_runs(self, thread_id: str) -> int: ...

    async def tokens_since(self, since: datetime) -> int:
        """Input + output tokens of runs started at or after `since` (timezone-aware)."""
        ...
