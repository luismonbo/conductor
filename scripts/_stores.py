"""What db.py reads the database through: a `Store` over one SQLite file or over the
API's Postgres. Both are read-only, and both return rows of the same shapes.

Imported as `_stores`: running a script puts scripts/ first on sys.path, and
pytest adds it through the pythonpath in pyproject.toml.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from harness.core.types import CHECKPOINT_TYPES

from _common import ScriptError, connect_postgres


@dataclass(frozen=True)
class Conversation:
    messages: list[Any]
    last_activity: str | None  # the latest checkpoint's timestamp


def _connect_ro(path: Path) -> sqlite3.Connection:
    # mode=ro: a missing file is an error rather than a new empty database.
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def _conversation(
    saver: SqliteSaver | PostgresSaver | None, thread_id: str
) -> Conversation | None:
    if saver is None:
        return None
    found = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if found is None:
        return None
    values = found.checkpoint.get("channel_values") or {}
    return Conversation(list(values.get("messages") or []), found.checkpoint.get("ts"))


class Store(Protocol):
    """What every command reads. Rows have SQLite's keys and value types, whichever
    database they came from: times as UTC ISO-8601 text, verdicts as 0/1, scores as
    JSON text."""

    def close(self) -> None: ...
    def thread_ids(self) -> list[str]: ...
    def run_thread_ids(self) -> list[str]: ...
    def conversation(self, thread_id: str) -> Conversation | None: ...
    def runs(self, thread_id: str | None = None) -> list[dict[str, Any]]: ...
    def guard_decisions(
        self,
        *,
        thread_id: str | None = None,
        blocked_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...
    def runs_per_day(self, since_day: str) -> dict[str, dict[str, int]]: ...
    def blocked_per_day(self, since_day: str) -> dict[str, int]: ...
    def run_totals(self) -> tuple[int, int]: ...


class SqliteStore:
    """Read-only access to one SQLite file."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise ScriptError(
                f"no database at {path} (run from the repo root, or pass --db PATH)"
            )
        self._sql = _connect_ro(path)
        self._sql.row_factory = sqlite3.Row
        tables = self._sql.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        self._tables = {row["name"] for row in tables}
        self._saver_conn: sqlite3.Connection | None = None
        self._saver: SqliteSaver | None = None
        # Without its tables the saver's setup would try to create them.
        if {"checkpoints", "writes"} <= self._tables:
            self._saver_conn = _connect_ro(path)
            # Reads happen outside a graph, so strict mode (LANGGRAPH_STRICT_MSGPACK)
            # would return the harness's own types as plain dicts unless allowed here.
            serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
            self._saver = SqliteSaver(self._saver_conn, serde=serde)

    def close(self) -> None:
        self._sql.close()
        if self._saver_conn is not None:
            self._saver_conn.close()

    def thread_ids(self) -> list[str]:
        """Threads with a conversation, newest first: checkpoint ids sort by time."""
        if "checkpoints" not in self._tables:
            return []
        rows = self._sql.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id "
            "ORDER BY MAX(checkpoint_id) DESC"
        )
        return [row["thread_id"] for row in rows]

    def run_thread_ids(self) -> list[str]:
        if "runs" not in self._tables:
            return []
        rows = self._sql.execute(
            "SELECT DISTINCT thread_id FROM runs WHERE thread_id IS NOT NULL"
        )
        return [row["thread_id"] for row in rows]

    def conversation(self, thread_id: str) -> Conversation | None:
        return _conversation(self._saver, thread_id)

    def runs(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        """Run rows in start order: all of them, or one thread's."""
        if "runs" not in self._tables:
            return []
        if thread_id is None:
            rows = self._sql.execute("SELECT * FROM runs ORDER BY started_at")
        else:
            rows = self._sql.execute(
                "SELECT * FROM runs WHERE thread_id = ? ORDER BY started_at", (thread_id,)
            )
        return [dict(row) for row in rows]

    def guard_decisions(
        self,
        *,
        thread_id: str | None = None,
        blocked_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Guard rows, newest first."""
        if "guard_decisions" not in self._tables:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if blocked_only:
            clauses.append("allowed = 0")
        sql = "SELECT * FROM guard_decisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(row) for row in self._sql.execute(sql, params)]

    def runs_per_day(self, since_day: str) -> dict[str, dict[str, int]]:
        """Runs, threads and tokens per UTC day, from `since_day` (YYYY-MM-DD) on."""
        if "runs" not in self._tables:
            return {}
        rows = self._sql.execute(
            "SELECT substr(started_at, 1, 10) AS day, COUNT(*) AS runs, "
            "COUNT(DISTINCT thread_id) AS threads, "
            "SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS tokens "
            "FROM runs WHERE started_at >= ? GROUP BY day",
            (since_day,),
        )
        return {
            row["day"]: {
                "runs": row["runs"],
                "threads": row["threads"],
                "tokens": row["tokens"] or 0,
            }
            for row in rows
        }

    def blocked_per_day(self, since_day: str) -> dict[str, int]:
        if "guard_decisions" not in self._tables:
            return {}
        rows = self._sql.execute(
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS blocked "
            "FROM guard_decisions WHERE allowed = 0 AND created_at >= ? GROUP BY day",
            (since_day,),
        )
        return {row["day"]: row["blocked"] for row in rows}

    def run_totals(self) -> tuple[int, int]:
        """All-time runs, and their input + output tokens."""
        if "runs" not in self._tables:
            return 0, 0
        row = self._sql.execute(
            "SELECT COUNT(*) AS runs, "
            "SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS tokens "
            "FROM runs"
        ).fetchone()
        return row["runs"], row["tokens"] or 0


# The tables PostgresSaver reads a conversation from; it doesn't create them.
_POSTGRES_SAVER_TABLES = {"checkpoints", "checkpoint_blobs", "checkpoint_writes"}


def _utc_day_start(day: str) -> datetime:
    """Midnight UTC at the start of `day` (YYYY-MM-DD)."""
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)


def _sqlite_shaped(row: dict[str, Any]) -> dict[str, Any]:
    """A Postgres row with the value types SQLite returns, so every command prints
    the same from either database."""
    shaped: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            value = value.astimezone(timezone.utc).isoformat()
        elif isinstance(value, bool):
            value = int(value)
        elif isinstance(value, (dict, list)):
            value = json.dumps(value)
        shaped[key] = value
    return shaped


class PostgresStore:
    """Read-only access to the API's Postgres database (HARNESS_CHECKPOINTER=postgres)."""

    def __init__(self, dsn: str) -> None:
        self._conn = connect_postgres(dsn)
        try:
            # Every statement after this one runs in a read-only transaction.
            self._conn.execute("SET default_transaction_read_only = on")
            rows = self._conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema()"
            )
            self._tables = {row["table_name"] for row in rows}
            self._saver: PostgresSaver | None = None
            if _POSTGRES_SAVER_TABLES <= self._tables:
                serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
                self._saver = PostgresSaver(self._conn, serde=serde)
        except BaseException:
            # main() closes only a store it got back, and this one never gets out.
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def thread_ids(self) -> list[str]:
        """Threads with a conversation, newest first: checkpoint ids sort by time."""
        if "checkpoints" not in self._tables:
            return []
        rows = self._conn.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id "
            "ORDER BY MAX(checkpoint_id) DESC"
        )
        return [row["thread_id"] for row in rows]

    def run_thread_ids(self) -> list[str]:
        if "runs" not in self._tables:
            return []
        rows = self._conn.execute(
            "SELECT DISTINCT thread_id FROM runs WHERE thread_id IS NOT NULL"
        )
        return [row["thread_id"] for row in rows]

    def conversation(self, thread_id: str) -> Conversation | None:
        return _conversation(self._saver, thread_id)

    def runs(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        """Run rows in start order: all of them, or one thread's."""
        if "runs" not in self._tables:
            return []
        if thread_id is None:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY started_at")
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE thread_id = %s ORDER BY started_at", (thread_id,)
            )
        return [_sqlite_shaped(row) for row in rows]

    def guard_decisions(
        self,
        *,
        thread_id: str | None = None,
        blocked_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Guard rows, newest first."""
        if "guard_decisions" not in self._tables:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = %s")
            params.append(thread_id)
        if blocked_only:
            clauses.append("NOT allowed")
        sql = "SELECT * FROM guard_decisions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        return [_sqlite_shaped(row) for row in self._conn.execute(sql, params)]

    def runs_per_day(self, since_day: str) -> dict[str, dict[str, int]]:
        """Runs, threads and tokens per UTC day, from `since_day` (YYYY-MM-DD) on."""
        if "runs" not in self._tables:
            return {}
        rows = self._conn.execute(
            "SELECT to_char(started_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, "
            "COUNT(*) AS runs, COUNT(DISTINCT thread_id) AS threads, "
            "SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS tokens "
            "FROM runs WHERE started_at >= %s GROUP BY day",
            (_utc_day_start(since_day),),
        )
        return {
            row["day"]: {
                "runs": row["runs"],
                "threads": row["threads"],
                "tokens": row["tokens"] or 0,
            }
            for row in rows
        }

    def blocked_per_day(self, since_day: str) -> dict[str, int]:
        if "guard_decisions" not in self._tables:
            return {}
        rows = self._conn.execute(
            "SELECT to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, "
            "COUNT(*) AS blocked "
            "FROM guard_decisions WHERE NOT allowed AND created_at >= %s GROUP BY day",
            (_utc_day_start(since_day),),
        )
        return {row["day"]: row["blocked"] for row in rows}

    def run_totals(self) -> tuple[int, int]:
        """All-time runs, and their input + output tokens."""
        if "runs" not in self._tables:
            return 0, 0
        row = self._conn.execute(
            "SELECT COUNT(*) AS runs, "
            "SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)) AS tokens "
            "FROM runs"
        ).fetchone()
        return (row["runs"], row["tokens"] or 0) if row else (0, 0)
