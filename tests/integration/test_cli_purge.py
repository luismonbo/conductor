"""harness.cli.purge: a pass through real storage (SQLite and Postgres), main's exit
codes and log line, and the settings the production job boots with."""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

import harness.cli.purge as purge_cli
from harness.adapters.llm.fake import FakeLLMClient
from harness.config.settings import Settings, get_settings
from harness.core.guard import GuardDecision
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.observability.run_store import RowCounts
from harness.orchestration.loop import build_graph
from harness.orchestration.retention import PurgeResult
from harness.orchestration.storage import open_storage
from tests.conftest import postgres_schema

DAY = timedelta(days=1)
SID = "session-a"
_TARGETS = Path(__file__).resolve().parents[2] / "config" / "targets"


def _settings(checkpointer: str, url: str, retention_days: int = 30) -> Settings:
    return Settings(
        _env_file=None, api_key="test-key", input_guard_enabled=False,
        checkpointer=checkpointer, checkpointer_url=url, retention_days=retention_days,
    )


@pytest.fixture(params=["sqlite", "postgres"])
def purge_settings(request, tmp_path) -> Iterator[Settings]:
    if request.param == "sqlite":
        yield _settings("sqlite", str(tmp_path / "harness.sqlite"))
    else:
        with postgres_schema() as dsn:  # skips without Postgres
            yield _settings("postgres", dsn)


async def _converse(checkpointer, thread_id: str) -> None:
    """One real turn through the graph, so the thread has real checkpoints."""
    graph = build_graph(
        llm=FakeLLMClient([LLMResponse(text="Hi there.")]),
        checkpointer=checkpointer,
        registry=ToolRegistry(),
        system_prompt="You are a test agent.",
    )
    state = {"messages": [Message(Role.USER, "hello")], "iteration": 0, "max_iterations": 2}
    config = {"configurable": {"thread_id": thread_id, "event_queue": asyncio.Queue()}}
    await graph.ainvoke(state, config)


async def _seed(settings: Settings, clock) -> tuple[str, str]:
    """An old conversation (at the clock's start) and one 40 days newer, as the API
    leaves them: checkpoints, a run and a guard row. Returns their thread ids."""
    old, new = str(uuid.uuid4()), str(uuid.uuid4())
    t0 = clock.now
    storage = await open_storage(settings)
    try:
        for thread_id, when in ((old, t0), (new, t0 + 40 * DAY)):
            clock.now = when
            await _converse(storage.checkpointer, thread_id)
            run_id = f"run-{thread_id}"
            await storage.run_store.start_run(run_id, thread_id, "default", "fake", SID)
            await storage.run_store.record_guard_decision(
                GuardDecision(allowed=True, stage="classifier"),
                session_id=SID, thread_id=thread_id, run_id=run_id, message="hello",
            )
    finally:
        clock.now = t0
        await storage.aclose()
    return old, new


async def _state(settings: Settings, thread_id: str) -> tuple[bool, bool]:
    """(has a checkpoint, has an ownership row), read through freshly opened storage."""
    storage = await open_storage(settings)
    try:
        saved = await storage.checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
        owned = await storage.run_store.owns_thread(SID, thread_id)
    finally:
        await storage.aclose()
    return saved is not None, owned


async def test_a_pass_deletes_the_old_conversation_only(purge_settings, run_store_clock):
    old, new = await _seed(purge_settings, run_store_clock)

    result = await purge_cli.run_purge(purge_settings, now=run_store_clock.now + 40 * DAY)

    assert (result.threads, result.failed_threads) == (1, 0)
    assert await _state(purge_settings, old) == (False, False)
    assert await _state(purge_settings, new) == (True, True)


def test_main_does_nothing_at_retention_days_0(monkeypatch):
    monkeypatch.setattr(
        purge_cli, "get_settings", lambda: _settings("sqlite", ":memory:", retention_days=0)
    )

    def must_not_open(settings):
        raise AssertionError("main opened storage at retention_days 0")

    monkeypatch.setattr(purge_cli, "open_storage", must_not_open)

    assert purge_cli.main([]) == 0


def test_main_exits_1_when_a_thread_fails(monkeypatch):
    monkeypatch.setattr(purge_cli, "get_settings", lambda: _settings("sqlite", ":memory:"))

    async def one_thread_failed(settings, *, now, dry_run=False):
        return PurgeResult(
            cutoff=now - 30 * DAY, dry_run=dry_run, threads=2, failed_threads=1,
            thread_rows=RowCounts(2, 2), unthreaded_rows=RowCounts(0, 0),
        )

    monkeypatch.setattr(purge_cli, "run_purge", one_thread_failed)

    assert purge_cli.main([]) == 1


def test_main_exits_1_when_the_pass_raises(monkeypatch, caplog):
    monkeypatch.setattr(purge_cli, "get_settings", lambda: _settings("sqlite", ":memory:"))

    async def database_down(settings, *, now, dry_run=False):
        raise RuntimeError("the database is down")

    monkeypatch.setattr(purge_cli, "run_purge", database_down)

    with caplog.at_level(logging.INFO, logger="harness"):
        assert purge_cli.main([]) == 1
    assert "purge failed" in caplog.text


def test_main_reports_counts_and_no_thread_id(purge_settings, run_store_clock, monkeypatch, caplog):
    old, new = asyncio.run(_seed(purge_settings, run_store_clock))
    monkeypatch.setattr(purge_cli, "get_settings", lambda: purge_settings)
    monkeypatch.setattr(purge_cli, "_now", lambda: run_store_clock.now + 40 * DAY)

    with caplog.at_level(logging.INFO, logger="harness"):
        assert purge_cli.main(["--dry-run"]) == 0
        assert "purge (dry run): cutoff=" in caplog.text
        assert "threads=1 " in caplog.text
        assert asyncio.run(_state(purge_settings, old)) == (True, True)

        assert purge_cli.main([]) == 0
    assert "purge: cutoff=" in caplog.text
    assert "failed_threads=0" in caplog.text
    assert asyncio.run(_state(purge_settings, old)) == (False, False)
    assert asyncio.run(_state(purge_settings, new)) == (True, True)
    assert old not in caplog.text
    assert new not in caplog.text


def test_the_command_logs_its_line(tmp_path, monkeypatch):
    # Run with `python -m`, the module is __main__: its lines must still reach stderr.
    monkeypatch.setenv("HARNESS_RETENTION_DAYS", "0")
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "false")
    monkeypatch.delenv("HARNESS_TARGET", raising=False)

    done = subprocess.run(
        [sys.executable, "-m", "harness.cli.purge", "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,  # tmp_path: no .env
    )

    assert done.returncode == 0, done.stderr
    assert "purge: retention_days is 0, nothing to delete" in done.stderr


def test_the_job_environment_builds_settings(tmp_path, monkeypatch):
    """The production job: the website target, its DSN, auth off, and no API key."""
    targets = tmp_path / "config" / "targets"
    targets.mkdir(parents=True)
    shutil.copy(_TARGETS / "luismonbo-com.yaml", targets)
    monkeypatch.chdir(tmp_path)  # no .env here, as in the image
    monkeypatch.setenv("HARNESS_TARGET", "luismonbo-com")
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "false")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", "postgresql://u:p@example.invalid/harness")
    # The suite's own defaults would hide the target's values.
    for name in (
        "HARNESS_API_KEY", "HARNESS_INPUT_GUARD_ENABLED", "HARNESS_AGENT",
        "HARNESS_CHECKPOINTER", "HARNESS_RETENTION_DAYS", "HARNESS_WIKI_DIR",
        "HARNESS_WIKI_BLOB_ACCOUNT_URL", "HARNESS_WIKI_BLOB_CONTAINER", "HARNESS_WIKI_BLOB_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.retention_days == 30
    assert settings.checkpointer == "postgres"
    assert settings.auth_enabled is False
