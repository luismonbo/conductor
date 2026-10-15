"""open_pool's failures, which need no database: nothing answers on port 1."""
from __future__ import annotations

import logging

import pytest

import harness.adapters.postgres.pool as pool_module
from harness.adapters.postgres.pool import host_of, open_pool

_UNREACHABLE = "postgresql://harness_app:hunter2secret@127.0.0.1:1/harness"


async def test_an_unreachable_database_fails_naming_only_the_host(monkeypatch, caplog):
    # Review Focus 3: a wrong secret or Neon down stops the boot, and says where.
    monkeypatch.setattr(pool_module, "OPEN_TIMEOUT_SECONDS", 1)
    opened: list[pool_module.AsyncConnectionPool] = []

    class _Recorded(pool_module.AsyncConnectionPool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(pool_module, "AsyncConnectionPool", _Recorded)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConnectionError, match="127.0.0.1 within 1 s") as excinfo:
            await open_pool(_UNREACHABLE)

    assert "hunter2secret" not in str(excinfo.value)
    assert "hunter2secret" not in caplog.text
    assert opened[0].closed  # no reconnection attempts left running


def test_host_of_names_only_the_host():
    neon = (
        "postgresql://harness_app:hunter2secret@ep-x.eu-central-1.aws.neon.tech/harness"
        "?sslmode=verify-full"
    )
    assert host_of(neon) == "ep-x.eu-central-1.aws.neon.tech"
    assert host_of("postgresql://harness_app@/harness") == "localhost"
    assert "hunter2secret" not in host_of("password=hunter2secret host")  # unreadable
