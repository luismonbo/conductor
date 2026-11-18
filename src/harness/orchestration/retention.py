"""Retention: delete what a conversation left behind once it is retention_days old.

One pass (`purge`) finds the stale threads in the run store, deletes each one's
checkpoints and then its runs and guard_decisions rows, and then deletes the old rows
that belong to no thread (blocked first messages, legacy runs). harness.cli.purge runs
a pass; in production a scheduled Container Apps Job runs it daily. Design:
docs/superpowers/specs/2026-10-14-retention-purge-design.md (§3, §4.3, §8).

Checkpoints go first. The run-store rows are how the purge finds a thread, so if the
checkpoint delete fails, the rows stay and the next pass retries the thread. The other
order could leave checkpoints that nothing points to, which no pass would ever find.

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
    failed_threads: int  # stale threads whose deletion failed (always 0 in a dry run)
    thread_rows: RowCounts  # rows deleted with those threads; none in a dry run
    unthreaded_rows: RowCounts  # rows with no thread before cutoff: deleted, or found


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
    skipped; errors that stop the whole pass, like a failed query for the stale
    threads, propagate.
    """
    if retention_days < 1:
        raise ValueError(f"retention_days must be at least 1, not {retention_days}")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    cutoff = now - timedelta(days=retention_days)
    stale = await run_store.stale_threads(cutoff)
    if dry_run:
        return PurgeResult(
            cutoff=cutoff, dry_run=True, threads=len(stale), failed_threads=0,
            thread_rows=_NONE, unthreaded_rows=await run_store.count_unthreaded_before(cutoff),
        )

    deleted, failed, runs, decisions = 0, 0, 0, 0
    for thread_id in stale:
        rows = await _delete_thread(checkpointer, run_store, thread_id, cutoff)
        if rows is None:
            failed += 1
            continue
        deleted += 1
        runs += rows.runs
        decisions += rows.guard_decisions
    return PurgeResult(
        cutoff=cutoff, dry_run=False, threads=deleted, failed_threads=failed,
        thread_rows=RowCounts(runs=runs, guard_decisions=decisions),
        unthreaded_rows=await run_store.delete_unthreaded_before(cutoff),
    )


async def _delete_thread(
    checkpointer: BaseCheckpointSaver, run_store: RunStore, thread_id: str, cutoff: datetime
) -> RowCounts | None:
    """Delete one thread, checkpoints first; None if either step failed.

    Rows from `cutoff` on survive: a visitor who writes into the thread while the pass
    runs keeps it, and a later pass finds it, and the checkpoints that turn writes,
    through those rows. Deleting them too would leave those checkpoints unreachable.
    The error is logged by type only: its message could carry the thread id.
    """
    try:
        await checkpointer.adelete_thread(thread_id)
    except Exception as exc:
        logger.error("purge: deleting a thread's checkpoints failed (%s)", type(exc).__name__)
        return None
    try:
        return await run_store.delete_thread_rows(thread_id, cutoff)
    except Exception as exc:
        logger.error("purge: deleting a thread's rows failed (%s)", type(exc).__name__)
        return None
