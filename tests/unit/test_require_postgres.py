"""require_postgres: an unreachable Postgres skips its tests locally, and fails them in CI."""
from __future__ import annotations

import pytest

import tests.conftest as suite

# Port 1 on 127.0.0.1: refused at once.
_UNREACHABLE = "postgresql://svc_user:SECRET@127.0.0.1:1/appdb"


@pytest.fixture
def unreachable_postgres(monkeypatch):
    """The suite's Postgres pointed at port 1. The probe's answer is cached for the
    whole run, so clear it on both sides: this test's answer must not outlive it, and
    the next Postgres test must probe the real server."""
    monkeypatch.setattr(suite, "POSTGRES_DSN", _UNREACHABLE)
    suite._postgres_unreachable.cache_clear()
    yield
    suite._postgres_unreachable.cache_clear()


def _outcome() -> BaseException:
    """What require_postgres() ends the test with: a skip, or a failure."""
    try:
        suite.require_postgres()
    except (pytest.skip.Exception, pytest.fail.Exception) as outcome:
        return outcome
    raise AssertionError("require_postgres() let the test run")


@pytest.mark.parametrize(
    ("required", "expected"),
    [
        pytest.param(None, pytest.skip.Exception, id="locally-it-skips"),
        pytest.param("0", pytest.skip.Exception, id="0-is-not-required"),
        # CI sets it: a Postgres service that never came up must not turn every
        # Postgres test into a quiet skip behind a green run.
        pytest.param("1", pytest.fail.Exception, id="in-ci-it-fails"),
    ],
)
def test_an_unreachable_postgres_skips_unless_it_is_required(
    unreachable_postgres, monkeypatch, required, expected
):
    if required is None:
        monkeypatch.delenv("HARNESS_TEST_POSTGRES_REQUIRED", raising=False)
    else:
        monkeypatch.setenv("HARNESS_TEST_POSTGRES_REQUIRED", required)

    outcome = _outcome()

    assert type(outcome) is expected
    message = str(outcome)
    assert "127.0.0.1" in message  # the host, so the log says where
    assert "SECRET" not in message and "postgresql://" not in message  # never the DSN
