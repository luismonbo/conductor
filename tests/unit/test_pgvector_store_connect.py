"""PgVectorStore's connects, which need no database: psycopg's connect is replaced by a
recorder that raises, so a store that reaches it fails the test."""
from __future__ import annotations

import logging

import psycopg
import pytest

from harness.adapters.vectorstore.pgvector_store import PgVectorStore

# Passwords with an unencoded "%" and "@" (the same two as test_postgres_pool_open.py).
_MALFORMED = [
    pytest.param("postgresql://svc_user:ab%zzSECRET@127.0.0.1:1/appdb", id="unencoded-percent"),
    pytest.param("postgresql://svc_user:ab@cdSECRET@127.0.0.1:1/appdb", id="unencoded-at"),
]

# Every public method connects through _connect, so search stands for them all; drop
# is the one that connected apart from it.
_CONNECTING_CALLS = [
    pytest.param(lambda store: store.search([1.0, 0.0, 0.0], k=1), id="search"),
    pytest.param(lambda store: store.drop(), id="drop"),
]


@pytest.fixture
def connect_attempts(monkeypatch) -> list[str]:
    """One entry per psycopg.AsyncConnection.connect call, each of which raises.

    The entries hold no connection string, so a failed assertion can't print one.
    """
    attempts: list[str] = []

    async def refuse(*args, **kwargs):
        attempts.append("connect")
        raise AssertionError("psycopg.AsyncConnection.connect was reached")

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", refuse)
    return attempts


@pytest.mark.parametrize("dsn", _MALFORMED)
@pytest.mark.parametrize("call", _CONNECTING_CALLS)
async def test_a_malformed_url_is_refused_before_the_store_connects_or_logs(
    call, dsn, connect_attempts, caplog
):
    # An unencoded "%" makes psycopg's parser quote the whole password in its error; an
    # unencoded "@" makes libpq read part of it as a host, which its error then names.
    store = PgVectorStore(dsn=dsn)  # construction checks nothing: it runs at startup
    # psycopg pins its own logger to WARNING.
    caplog.set_level(logging.DEBUG, logger="psycopg")
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ValueError) as excinfo:
        await call(store)

    assert connect_attempts == []
    message = str(excinfo.value)
    for fragment in ("SECRET", "svc_user", "127.0.0.1", "appdb"):
        assert fragment not in message
        assert fragment not in caplog.text
    # A traceback prints the whole chain, and the parser's error echoes the token.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
