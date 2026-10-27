"""The test Postgres: each test's tables land in a schema of its own, dropped after."""
from __future__ import annotations

import psycopg

from tests.conftest import POSTGRES_DSN, postgres_schema


def test_a_test_schema_holds_the_tables_and_goes_away():
    with postgres_schema() as dsn:
        assert dsn.startswith("postgresql://")
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE TABLE runs (run_id TEXT)")
            [(schema,)] = conn.execute(
                "SELECT table_schema FROM information_schema.tables "
                "WHERE table_name = 'runs' AND table_schema = current_schema()"
            ).fetchall()
    assert schema.startswith("test_")
    with psycopg.connect(POSTGRES_DSN) as conn:
        gone = conn.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s", (schema,)
        ).fetchone()
    assert gone is None


def test_options_reach_the_session():
    with postgres_schema("-c TimeZone=America/New_York") as dsn:
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SHOW TimeZone").fetchone() == ("America/New_York",)
