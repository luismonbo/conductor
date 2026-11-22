"""Retention: delete what a conversation left behind once it is retention_days old.

One pass (`purge`) finds the stale threads in the run store, deletes each one's
checkpoints and then its runs and guard_decisions rows, then deletes the threads no row
names (unrecorded: both of their best-effort run-store writes failed) by the time of
their newest checkpoint, and then the old rows that belong to no thread (blocked first
messages, legacy runs). harness.cli.purge runs a pass; in production a scheduled
Container Apps Job runs it daily. Design:
docs/superpowers/specs/2026-10-14-retention-purge-design.md (§3, §4.3, §8).

Checkpoints go first. The run-store rows are how the purge finds a thread, so if the
checkpoint delete fails, the rows stay and the next pass retries the thread. The other
order could leave checkpoints that no row names, which a later pass would find only as
an unrecorded thread, dated by its checkpoints instead.

Logs carry counts only: never a thread id, a session id or message text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from harness.observability.run_store import RowCounts, RunStore

# Annotations only, as in orchestration/storage.py: importing this module doesn't
# load LangGraph.
if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

_NONE = RowCounts(runs=0, guard_decisions=0)


@dataclass(frozen=True)
class PurgeResult:
    """What one pass deleted, or in a dry run, found."""

    cutoff: datetime
    dry_run: bool
    threads: int  # stale threads deleted; in a dry run, found
    failed_threads: int  # threads, stale or unrecorded, whose deletion failed (0 in a dry run)
    thread_rows: RowCounts  # rows deleted with those threads; none in a dry run
    unthreaded_rows: RowCounts  # rows with no thread before cutoff: deleted, or found
    # True when deleting the rows with no thread failed; they wait for the next pass.
    unthreaded_failed: bool = False
    # Unrecorded threads (no row names them) whose newest checkpoint is from before
    # cutoff: deleted, or in a dry run, found.
    unrecorded_threads: int = 0

    @property
    def complete(self) -> bool:
        """True when nothing the pass tried to delete failed."""
        return self.failed_threads == 0 and not self.unthreaded_failed


async def purge(
    checkpointer: BaseCheckpointSaver,
    run_store: RunStore,
    *,
    now: datetime,
    retention_days: int,
    dry_run: bool = False,
) -> PurgeResult:
    """Delete every conversation whose latest activity is retention_days before `now`.

    Refuses retention_days below 1 (0 means "keep everything", and as a cutoff it
    would be `now`) and a naive `now` (compared with stored UTC times, it would shift
    the cutoff by the machine's offset). A thread whose deletion fails is counted and
    skipped; if deleting the rows with no thread fails, the result says so and still
    counts the threads that went. Errors that stop the whole pass, like a failed query
    for the stale threads, propagate.
    """
    if retention_days < 1:
        raise ValueError(f"retention_days must be at least 1, not {retention_days}")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    cutoff = now - timedelta(days=retention_days)
    stale = await run_store.stale_threads(cutoff)
    unrecorded = await _due_unrecorded(checkpointer, run_store, cutoff)
    if dry_run:
        return PurgeResult(
            cutoff=cutoff, dry_run=True, threads=len(stale), failed_threads=0,
            thread_rows=_NONE, unthreaded_rows=await run_store.count_unthreaded_before(cutoff),
            unrecorded_threads=len(unrecorded),
        )

    deleted, failed, rows = await _delete_stale(checkpointer, run_store, stale, cutoff)
    # True for each unrecorded thread whose checkpoints went.
    gone = [await _delete_checkpoints(checkpointer, thread_id) for thread_id in unrecorded]
    unthreaded = await _delete_unthreaded(run_store, cutoff)
    return PurgeResult(
        cutoff=cutoff, dry_run=False, threads=deleted,
        failed_threads=failed + gone.count(False), thread_rows=rows,
        unthreaded_rows=_NONE if unthreaded is None else unthreaded,
        unthreaded_failed=unthreaded is None, unrecorded_threads=gone.count(True),
    )


async def _due_unrecorded(
    checkpointer: BaseCheckpointSaver, run_store: RunStore, cutoff: datetime
) -> list[str]:
    """The unrecorded threads whose newest checkpoint is from before `cutoff`.

    No row names them, so that checkpoint's time (LangGraph writes it in UTC) is their
    last activity, and nothing can add to them meanwhile: the API checks ownership
    through runs rows, which they lack. A thread with no checkpoint in the root
    namespace has nothing to date it by, and stays.
    """
    due = []
    for thread_id in await run_store.unrecorded_threads():
        latest = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        if latest is not None and datetime.fromisoformat(latest.checkpoint["ts"]) < cutoff:
            due.append(thread_id)
    return due


async def _delete_stale(
    checkpointer: BaseCheckpointSaver, run_store: RunStore, stale: list[str], cutoff: datetime
) -> tuple[int, int, RowCounts]:
    """Delete each stale thread: (deleted, failed, the rows that went with them)."""
    deleted, failed, runs, decisions = 0, 0, 0, 0
    for thread_id in stale:
        rows = await _delete_thread(checkpointer, run_store, thread_id, cutoff)
        if rows is None:
            failed += 1
            continue
        deleted += 1
        runs += rows.runs
        decisions += rows.guard_decisions
    return deleted, failed, RowCounts(runs=runs, guard_decisions=decisions)


async def _delete_unthreaded(run_store: RunStore, cutoff: datetime) -> RowCounts | None:
    """Delete the old rows with no thread; None if that failed.

    Caught rather than raised, so the counts of the threads already deleted still
    reach the log. One transaction, so a failure deleted none of them.
    """
    try:
        return await run_store.delete_unthreaded_before(cutoff)
    except Exception as exc:
        logger.error("purge: deleting the rows with no thread failed (%s)", type(exc).__name__)
        return None


async def _delete_thread(
    checkpointer: BaseCheckpointSaver, run_store: RunStore, thread_id: str, cutoff: datetime
) -> RowCounts | None:
    """Delete one thread, checkpoints first; None if either step failed.

    Rows from `cutoff` on survive: a visitor who writes into the thread while the pass
    runs keeps it, and a later pass finds it, and the checkpoints that turn writes,
    through those rows. Deleting them too would leave those checkpoints unreachable.
    The error is logged by type only: its message could carry the thread id.
    """
    if not await _delete_checkpoints(checkpointer, thread_id):
        return None
    try:
        return await run_store.delete_thread_rows(thread_id, cutoff)
    except Exception as exc:
        logger.error("purge: deleting a thread's rows failed (%s)", type(exc).__name__)
        return None


async def _delete_checkpoints(checkpointer: BaseCheckpointSaver, thread_id: str) -> bool:
    """Delete one thread's checkpoints; False if that failed, logged by type only."""
    try:
        await checkpointer.adelete_thread(thread_id)
    except Exception as exc:
        logger.error("purge: deleting a thread's checkpoints failed (%s)", type(exc).__name__)
        return False
    return True
