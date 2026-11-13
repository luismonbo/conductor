"""Suite-wide test defaults."""
from __future__ import annotations

import os

# Run the suite as production runs: LangGraph loads only its built-in safe types and the
# ones a graph's state declares (the Dockerfile sets the same variable). LangGraph reads
# the variable when it is first imported, so this comes before every other import, and
# pytest imports this file before the conftests below it (tests/integration/conftest.py
# imports the API, which loads LangGraph). test_checkpoint_serde.py fails if it stops
# taking effect.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

import functools
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote

import psycopg
import pytest
from psycopg import sql

from harness.adapters.postgres.pool import host_of
from harness.observability.logging_setup import _PRIVATE_AT_DEBUG, _VISITOR_IP_AT_WARNING

# The loggers whose levels configure_logging (run by the app's startup) sets.
_CONFIGURED_LOGGERS = ("harness", *_PRIVATE_AT_DEBUG, *_VISITOR_IP_AT_WARNING)


@pytest.fixture(autouse=True)
def _restore_logger_levels():
    """Logger levels are process-wide: put back the ones the app's startup sets,
    so no test inherits another test's logging setup."""
    saved = {name: logging.getLogger(name).level for name in _CONFIGURED_LOGGERS}
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture(autouse=True)
def _input_guard_off_by_default(monkeypatch):
    """The input guard is default-on and refuses to start without an Azure
    endpoint, so every test runs with it off. Guard tests opt back in explicitly
    (delenv this, or pass input_guard_enabled=True with an endpoint)."""
    monkeypatch.setenv("HARNESS_INPUT_GUARD_ENABLED", "false")


@pytest.fixture
def fixture_agents(monkeypatch):
    """Resolve agent names under tests/fixtures/agents instead of src/harness/agents."""
    import harness.orchestration.build as build_module
    from harness.orchestration.agents import load_agent

    monkeypatch.setattr(
        build_module,
        "load_agent",
        lambda name: load_agent(name, package="tests.fixtures.agents"),
    )


_WIKI_ENV = (
    "HARNESS_WIKI_DIR",
    "HARNESS_WIKI_BLOB_ACCOUNT_URL",
    "HARNESS_WIKI_BLOB_CONTAINER",
    "HARNESS_WIKI_BLOB_PREFIX",
)


@pytest.fixture(autouse=True)
def _no_wiki_by_default(monkeypatch):
    """Every test starts with no wiki source, whatever the developer's .env says.
    Tests that need a wiki set HARNESS_WIKI_DIR (or pass wiki_dir) themselves."""
    for key in _WIKI_ENV:
        monkeypatch.setenv(key, "")


# --- Postgres ------------------------------------------------------------------
# The server the Postgres tests use: `make infra` locally, a service container in
# CI. A postgresql:// URL: the API's settings accept only that form. Without it the
# Postgres tests skip, unless HARNESS_TEST_POSTGRES_REQUIRED=1 (see require_postgres).
POSTGRES_DSN = os.environ.get(
    "HARNESS_TEST_POSTGRES_DSN", "postgresql://harness:harness@localhost:5432/harness"
)


@functools.cache
def _postgres_unreachable() -> str | None:
    """Why the test Postgres can't be reached, or None. Checked once per run."""
    try:
        psycopg.connect(POSTGRES_DSN, connect_timeout=3).close()
    except psycopg.OperationalError as exc:
        return str(exc)
    return None


def require_postgres() -> None:
    """Skip the calling test unless the test Postgres answers.

    With HARNESS_TEST_POSTGRES_REQUIRED=1 (CI sets it) the test fails instead: a
    Postgres service that never came up must not turn every Postgres test into a
    quiet skip behind a green run. Either message names the host, never the DSN.
    """
    reason = _postgres_unreachable()
    if reason is None:
        return
    message = f"Postgres unreachable at {host_of(POSTGRES_DSN)}: {reason}"
    if os.environ.get("HARNESS_TEST_POSTGRES_REQUIRED") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def with_options(dsn: str, options: str) -> str:
    """`dsn`, a postgresql:// URL, with libpq `options` such as "-c search_path=x"."""
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}options={quote(options)}"


@contextmanager
def postgres_schema(options: str = "") -> Iterator[str]:
    """A DSN whose tables land in a fresh schema, dropped on exit.

    `options` adds session settings, e.g. "-c TimeZone=America/New_York".
    """
    require_postgres()
    schema = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(POSTGRES_DSN, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield with_options(POSTGRES_DSN, f"-c search_path={schema} {options}".strip())
    finally:
        with psycopg.connect(POSTGRES_DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def postgres_dsn() -> Iterator[str]:
    """A DSN on the test Postgres, in a schema of this test's own."""
    with postgres_schema() as dsn:
        yield dsn
