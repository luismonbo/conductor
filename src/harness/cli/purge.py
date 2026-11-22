"""One retention pass: `python -m harness.cli.purge` (`make purge`; the production job).

Deletes every conversation whose latest activity is more than HARNESS_RETENTION_DAYS
old: its checkpoints, runs and guard_decisions rows, the conversations no row names (by
their newest checkpoint), and the old rows that belong to no thread
(orchestration/retention.py). Does nothing at retention_days 0.
`--dry-run` counts what a pass would delete and deletes nothing.

Logs one line of counts, never a thread id, a session id or message text: `purge:
cutoff=...` at INFO when everything it found went (the production alert watches for
that line), `purge incomplete: cutoff=...` at ERROR when part of it failed. Exit
code: 0 when the pass ran or had nothing to do; 1 when part of the pass failed or it
could not run, so a scheduler shows the run as failed and the next one retries.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from harness.config.settings import Settings, get_settings
from harness.observability.logging_setup import configure_logging
from harness.orchestration.retention import PurgeResult, purge
from harness.orchestration.storage import open_storage

# Named, not __name__: run with `python -m` the module is __main__, outside the
# "harness" logger that configure_logging opens to INFO, so every line would be dropped.
logger = logging.getLogger("harness.cli.purge")


def _now() -> datetime:
    """The pass's clock (tests replace it)."""
    return datetime.now(timezone.utc)


async def run_purge(settings: Settings, *, now: datetime, dry_run: bool = False) -> PurgeResult:
    """Open the storage `settings` names, run one pass, and close it again."""
    storage = await open_storage(settings)
    try:
        return await purge(
            storage.checkpointer, storage.run_store,
            now=now, retention_days=settings.retention_days, dry_run=dry_run,
        )
    finally:
        await storage.aclose()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli.purge",
        description="Delete the conversations older than HARNESS_RETENTION_DAYS.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="count what a pass would delete; delete nothing"
    )
    return parser.parse_args(argv)


def _log_result(result: PurgeResult) -> None:
    cutoff = result.cutoff.isoformat(timespec="seconds")
    if result.dry_run:
        logger.info(
            "purge (dry run): cutoff=%s threads=%d unrecorded_threads=%d unthreaded_runs=%d "
            "unthreaded_guard_decisions=%d",
            cutoff, result.threads, result.unrecorded_threads,
            result.unthreaded_rows.runs, result.unthreaded_rows.guard_decisions,
        )
        return
    counts = (
        "cutoff=%s threads=%d thread_runs=%d thread_guard_decisions=%d unrecorded_threads=%d "
        "unthreaded_runs=%d unthreaded_guard_decisions=%d failed_threads=%d"
    )
    values = (
        cutoff, result.threads,
        result.thread_rows.runs, result.thread_rows.guard_decisions, result.unrecorded_threads,
        result.unthreaded_rows.runs, result.unthreaded_rows.guard_decisions,
        result.failed_threads,
    )
    # Different words for each outcome: the production alert counts "purge: cutoff="
    # lines, so an incomplete pass must never log one.
    if result.complete:
        logger.info("purge: " + counts, *values)
    else:
        logger.error("purge incomplete: " + counts, *values)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.retention_days == 0:
        # 0 keeps everything, so there is nothing to open, let alone delete. A warning:
        # in the production job it means the setting was lost, and the alert, which
        # sees no "purge: cutoff=" line, fires.
        logger.warning("purge: retention_days is 0, nothing to delete")
        return 0
    try:
        result = asyncio.run(run_purge(settings, now=_now(), dry_run=args.dry_run))
    except Exception:
        logger.exception("purge failed")
        return 1
    _log_result(result)
    return 0 if result.complete else 1


if __name__ == "__main__":
    sys.exit(main())
