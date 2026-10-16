"""open_pool's failures, which need no database: nothing answers on port 1."""
from __future__ import annotations

import logging

import pytest

import harness.adapters.postgres.pool as pool_module
from harness.adapters.postgres.pool import check_dsn, host_of, open_pool

_UNREACHABLE = "postgresql://harness_app:hunter2secret@127.0.0.1:1/harness"

# Passwords with an unencoded "%" and "@". Port 1 on 127.0.0.1: a wrong fix that
# reaches libpq is refused in a moment.
_MALFORMED = [
    pytest.param("postgresql://svc_user:ab%zzSECRET@127.0.0.1:1/appdb", id="unencoded-percent"),
    pytest.param("postgresql://svc_user:ab@cdSECRET@127.0.0.1:1/appdb", id="unencoded-at"),
]


def _record_pools(monkeypatch) -> list[pool_module.AsyncConnectionPool]:
    """Every pool open_pool constructs, in order."""
    opened: list[pool_module.AsyncConnectionPool] = []

    class _Recorded(pool_module.AsyncConnectionPool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(pool_module, "AsyncConnectionPool", _Recorded)
    return opened


async def test_an_unreachable_database_fails_naming_only_the_host(monkeypatch, caplog):
    # Review Focus 3: a wrong secret or Neon down stops the boot, and says where.
    monkeypatch.setattr(pool_module, "OPEN_TIMEOUT_SECONDS", 1)
    opened = _record_pools(monkeypatch)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConnectionError, match="127.0.0.1 within 1 s") as excinfo:
            await open_pool(_UNREACHABLE)

    assert "hunter2secret" not in str(excinfo.value)
    assert "hunter2secret" not in caplog.text
    assert opened[0].closed  # no reconnection attempts left running


@pytest.mark.parametrize("dsn", _MALFORMED)
async def test_a_malformed_url_is_refused_before_anything_connects_or_logs(
    dsn, monkeypatch, caplog
):
    # Final review: an unencoded "%" made psycopg_pool log the whole password at
    # WARNING; an unencoded "@" made libpq read part of it as a host, which the
    # warning and the startup error then named.
    monkeypatch.setattr(pool_module, "OPEN_TIMEOUT_SECONDS", 1)
    opened = _record_pools(monkeypatch)
    # psycopg pins its own logger to WARNING; the pool logs under psycopg.pool.
    for name in ("psycopg", "psycopg.pool"):
        caplog.set_level(logging.DEBUG, logger=name)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ValueError) as excinfo:
        await open_pool(dsn)

    assert opened == []  # not even a pool: nothing connected
    message = str(excinfo.value)
    for fragment in ("SECRET", "svc_user", "127.0.0.1", "appdb"):
        assert fragment not in message
        assert fragment not in caplog.text
    # uvicorn prints a startup error's whole chain, and the parser's error echoes the token.
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


_WELL_FORMED = [
    pytest.param(
        "postgresql://harness_app:p%40ss%2Fw%25rd@ep-x.eu-central-1.aws.neon.tech/harness"
        "?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt",
        id="neon-with-encoded-password",
    ),
    pytest.param("postgres://harness@localhost/harness", id="no-password"),
    pytest.param("postgresql://harness:harness@localhost:5432/harness", id="local"),
    pytest.param("host=localhost user=harness password=p@ss/word dbname=harness", id="keywords"),
]


@pytest.mark.parametrize("dsn", _WELL_FORMED)
def test_check_dsn_lets_a_well_formed_string_through(dsn):
    check_dsn(dsn)  # does not raise


@pytest.mark.parametrize(
    "dsn",
    [
        *_MALFORMED,
        pytest.param("postgresql://svc_user:ab/cdSECRET@127.0.0.1:1/appdb", id="unencoded-slash"),
        pytest.param("postgresql://svc_user:SECRET@127.0.0.1:1/appdb?nope=1", id="unknown-option"),
        pytest.param("host=127.0.0.1 user=svc_user password=SECRET oops", id="unreadable-keywords"),
    ],
)
def test_check_dsn_refuses_a_string_libpq_would_misread_and_names_nothing_of_it(dsn):
    with pytest.raises(ValueError) as excinfo:
        check_dsn(dsn)

    for fragment in ("SECRET", "svc_user", "127.0.0.1", "appdb"):
        assert fragment not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_host_of_names_only_the_host():
    neon = (
        "postgresql://harness_app:hunter2secret@ep-x.eu-central-1.aws.neon.tech/harness"
        "?sslmode=verify-full"
    )
    assert host_of(neon) == "ep-x.eu-central-1.aws.neon.tech"
    assert host_of("postgresql://harness_app@/harness") == "localhost"
    assert "hunter2secret" not in host_of("password=hunter2secret host")  # unreadable
