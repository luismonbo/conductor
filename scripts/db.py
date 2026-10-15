"""Read-only views of the harness's conversations and run history.

Reads harness.sqlite: LangGraph's checkpoints (the conversations), `runs` (one
row per agent run: agent, tokens, stop reason) and `guard_decisions` (one row
per screened message). The file is opened read-only, so nothing here writes,
and a wrong path is an error instead of a new empty database. SQLite only for
now: a Postgres reader joins when the checkpointer and run store move there.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from harness.config.settings import Settings, get_settings

_EXAMPLES = """examples:
  uv run scripts/db.py threads
  uv run scripts/db.py thread 10fa
  uv run scripts/db.py thread 10fa --markdown > thread.md
  uv run scripts/db.py usage --days 14
  uv run scripts/db.py guard --blocked
"""

# The harness's own types inside checkpoints. Allowing them explicitly stops
# LangGraph's "unregistered type" warning on every read, and keeps reads working
# once LangGraph refuses unregistered types.
_HARNESS_TYPES = [("harness.core.types", name) for name in ("Message", "Role", "ToolCall")]

_TITLE_CHARS = 60
_CALL_CHARS = 150
_RESULT_CHARS = 120
_LABELS = {"user": "You", "assistant": "Assistant", "system": "System"}


class ScriptError(Exception):
    """A failure reported as one line on stderr, with exit code 1."""


# --- formatting --------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _short(value: str | None) -> str:
    """A UUID (thread, session or run id) as its first 8 characters."""
    return value[:8] if value else "-"


def _cut(text: str, width: int) -> str:
    """One line of at most `width` characters; newlines become spaces."""
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1] + "…"


def _fmt_time(iso: str | None) -> str:
    if not iso:
        return "-"
    moment = datetime.fromisoformat(iso).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M")


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _print_table(
    headers: list[str] | None,
    rows: list[list[str]],
    numeric: frozenset[int] = frozenset(),
    indent: str = "",
) -> None:
    """Aligned columns; `numeric` columns align right."""
    lines = [headers, *rows] if headers else rows
    if not lines:
        return
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    for line in lines:
        cells = [
            cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i])
            for i, cell in enumerate(line)
        ]
        print(indent + "  ".join(cells).rstrip())


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _count(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def _resolve(prefix: str, candidates: list[str], what: str) -> str:
    """The one candidate `prefix` names: an exact id, or a unique prefix of one."""
    if prefix in candidates:
        return prefix
    matches = sorted({c for c in candidates if c.startswith(prefix)})
    if not matches:
        raise ScriptError(f"no {what} matching {prefix!r}")
    if len(matches) > 1:
        listed = "\n  ".join(matches[:10])
        raise ScriptError(f"{prefix!r} matches {len(matches)} {what}s:\n  {listed}")
    return matches[0]


# --- messages ----------------------------------------------------------------
# Read by attribute, so a message in an unexpected shape prints as text instead
# of crashing the script.


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role) or "unknown")


def _content(message: Any) -> str:
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else str(message)


def _title(messages: list[Any]) -> str:
    """The first user message: the title the UI shows."""
    return next((_content(m) for m in messages if _role(m) == "user"), "")


def _tool_calls(message: Any) -> list[Any]:
    return list(getattr(message, "tool_calls", None) or ())


def _message_json(message: Any) -> dict[str, Any]:
    return {
        "role": _role(message),
        "content": _content(message),
        "tool_calls": [
            {
                "id": getattr(call, "id", None),
                "name": getattr(call, "name", None),
                "arguments": getattr(call, "arguments", None),
            }
            for call in _tool_calls(message)
        ],
        "tool_call_id": getattr(message, "tool_call_id", None),
        "name": getattr(message, "name", None),
    }


def _call_text(call: Any) -> str:
    """`name(key=value, …)`, with the values as JSON."""
    arguments = getattr(call, "arguments", None)
    if isinstance(arguments, dict):
        inner = ", ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in arguments.items()
        )
    else:
        inner = json.dumps(arguments, ensure_ascii=False, default=str)
    return f"{getattr(call, 'name', '?')}({inner})"


def _call_line(call: Any, full: bool) -> str:
    text = _call_text(call)
    return "  → " + (text if full else _cut(text, _CALL_CHARS))


def _result_lines(content: str, full: bool) -> list[str]:
    if full:
        first, *rest = content.splitlines() or [""]
        return [f"  ← {first}", *(f"    {line}" for line in rest)]
    line = " ".join(content.split())
    if len(line) <= _RESULT_CHARS:
        return [f"  ← {line}"]
    return [f"  ← {line[:_RESULT_CHARS]} … ({len(content):,} chars)"]


def _speaker_lines(label: str, text: str) -> list[str]:
    first, *rest = text.splitlines() or [""]
    return [f"{label}: {first}", *(f"  {line}" for line in rest)]


def _transcript_lines(messages: list[Any], *, full: bool, tools: bool) -> list[str]:
    lines: list[str] = []
    for message in messages:
        role = _role(message)
        if role == "tool":
            if tools:
                lines += _result_lines(_content(message), full)
            continue
        content = _content(message)
        calls = _tool_calls(message)
        if content or not calls:
            lines += _speaker_lines(_LABELS.get(role, role.capitalize()), content)
        if tools:
            lines += [_call_line(call, full) for call in calls]
    return lines


# --- reading -----------------------------------------------------------------


@dataclass(frozen=True)
class Conversation:
    messages: list[Any]
    last_activity: str | None  # the latest checkpoint's timestamp


@dataclass(frozen=True)
class RunSummary:
    runs: int
    input_tokens: int
    output_tokens: int
    agent: str | None  # the latest run's
    session_id: str | None  # the latest run's


def _summarize_runs(rows: list[dict[str, Any]]) -> dict[str, RunSummary]:
    """Per thread, from run rows in start order."""
    by_thread: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["thread_id"]:
            by_thread.setdefault(row["thread_id"], []).append(row)
    return {
        thread_id: RunSummary(
            runs=len(thread_rows),
            input_tokens=sum(row["input_tokens"] or 0 for row in thread_rows),
            output_tokens=sum(row["output_tokens"] or 0 for row in thread_rows),
            agent=thread_rows[-1]["agent"],
            # .get: databases from before visitor sessions have no session_id.
            session_id=thread_rows[-1].get("session_id"),
        )
        for thread_id, thread_rows in by_thread.items()
    }


def _connect_ro(path: Path) -> sqlite3.Connection:
    # mode=ro: a missing file is an error rather than a new empty database.
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


class Store:
    """Read-only access to one SQLite file; a Postgres version joins later."""

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
            serde = JsonPlusSerializer(allowed_msgpack_modules=_HARNESS_TYPES)
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
        if self._saver is None:
            return None
        found = self._saver.get_tuple({"configurable": {"thread_id": thread_id}})
        if found is None:
            return None
        values = found.checkpoint.get("channel_values") or {}
        return Conversation(list(values.get("messages") or []), found.checkpoint.get("ts"))

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


# --- commands ----------------------------------------------------------------


def _thread_info(
    thread_id: str, conversation: Conversation | None, summary: RunSummary | None
) -> dict[str, Any]:
    messages = conversation.messages if conversation else []
    return {
        "thread_id": thread_id,
        "last_activity": conversation.last_activity if conversation else None,
        "agent": summary.agent if summary else None,
        "session_id": summary.session_id if summary else None,
        "turns": summary.runs if summary else None,
        "input_tokens": summary.input_tokens if summary else None,
        "output_tokens": summary.output_tokens if summary else None,
        "messages": len(messages),
        "title": _title(messages),
    }


def cmd_threads(store: Store, args: argparse.Namespace, settings: Settings) -> None:
    summaries = _summarize_runs(store.runs())
    found: list[dict[str, Any]] = []
    for thread_id in store.thread_ids():
        summary = summaries.get(thread_id)
        if args.agent and (summary is None or summary.agent != args.agent):
            continue
        found.append(_thread_info(thread_id, store.conversation(thread_id), summary))
        if len(found) == args.limit:
            break
    if args.json:
        _print_json(found)
        return
    if not found:
        print("No threads.")
        return
    rows = []
    for info in found:
        has_runs = info["turns"] is not None
        tokens = info["input_tokens"] + info["output_tokens"] if has_runs else 0
        rows.append(
            [
                _short(info["thread_id"]),
                _fmt_time(info["last_activity"]),
                info["agent"] or "-",
                str(info["turns"]) if has_runs else "-",
                f"{tokens:,}" if has_runs else "-",
                _cut(info["title"], _TITLE_CHARS),
            ]
        )
    _print_table(
        ["thread", "last activity (UTC)", "agent", "turns", "tokens", "first message"],
        rows,
        numeric=frozenset({3, 4}),
    )


def _duration(row: dict[str, Any]) -> str:
    if not row["started_at"] or not row["finished_at"]:
        return "-"
    started = datetime.fromisoformat(row["started_at"])
    finished = datetime.fromisoformat(row["finished_at"])
    return f"{(finished - started).total_seconds():.1f}s"


def _verdict(row: dict[str, Any]) -> str:
    return "allowed" if row["allowed"] else "blocked"


def _guard_json(row: dict[str, Any]) -> dict[str, Any]:
    probabilities = row.get("probabilities")
    return {
        **row,
        "allowed": bool(row["allowed"]),
        "probabilities": json.loads(probabilities) if probabilities else None,
    }


def _print_thread(
    info: dict[str, Any],
    messages: list[Any],
    runs: list[dict[str, Any]],
    guard: list[dict[str, Any]],
    *,
    full: bool,
    tools: bool,
) -> None:
    tokens_in, tokens_out = info["input_tokens"] or 0, info["output_tokens"] or 0
    print(f"Thread {info['thread_id']}")
    print(
        " · ".join(
            [
                info["agent"] or "agent unknown",
                f"session {_short(info['session_id'])}",
                f"last activity {_fmt_time(info['last_activity'])} UTC",
                f"{info['turns'] or 0} turns",
                f"{tokens_in:,} in / {tokens_out:,} out tokens",
            ]
        )
    )
    print()
    if messages:
        print("\n".join(_transcript_lines(messages, full=full, tools=tools)))
    else:
        print("(no conversation stored)")
    if runs:
        print("\nRuns")
        _print_table(
            ["started (UTC)", "stop", "in", "out", "iterations", "duration"],
            [
                [
                    _fmt_time(row["started_at"]),
                    row["stopped_reason"] or "-",
                    _count(row["input_tokens"]),
                    _count(row["output_tokens"]),
                    _count(row["iterations"]),
                    _duration(row),
                ]
                for row in runs
            ],
            numeric=frozenset({2, 3, 4, 5}),
        )
    if guard:
        print("\nGuard")
        _print_table(
            ["time (UTC)", "verdict", "stage", "reason", "message"],
            [
                [
                    _fmt_time(row["created_at"]),
                    _verdict(row),
                    row["stage"],
                    row["reason"] or "-",
                    _cut(row["message_text"], _TITLE_CHARS),
                ]
                for row in guard
            ],
        )


def cmd_thread(store: Store, args: argparse.Namespace, settings: Settings) -> None:
    candidates = store.thread_ids() + store.run_thread_ids()
    thread_id = _resolve(args.thread_id, candidates, "thread")
    conversation = store.conversation(thread_id)
    runs = store.runs(thread_id)
    info = _thread_info(thread_id, conversation, _summarize_runs(runs).get(thread_id))
    messages = conversation.messages if conversation else []
    guard = list(reversed(store.guard_decisions(thread_id=thread_id)))  # oldest first
    if args.json:
        _print_json(
            {
                **info,
                "messages": [_message_json(message) for message in messages],
                "runs": runs,
                "guard_decisions": [_guard_json(row) for row in guard],
            }
        )
        return
    _print_thread(info, messages, runs, guard, full=args.full, tools=not args.no_tools)


# --- command line ------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db", help="the SQLite file (default: HARNESS_CHECKPOINTER_URL)"
    )
    parser = argparse.ArgumentParser(
        prog="db.py",
        description="Conversations, runs and guard decisions in harness.sqlite (read-only).",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    threads = commands.add_parser(
        "threads", parents=[common], help="conversations, newest first"
    )
    threads.add_argument("--limit", type=_positive_int, default=20)
    threads.add_argument(
        "--agent", help="only threads whose latest run used this agent"
    )
    threads.add_argument("--json", action="store_true", help="full rows as JSON")

    thread = commands.add_parser(
        "thread",
        parents=[common],
        help="one conversation, with its runs and guard decisions",
    )
    thread.add_argument("thread_id", help="a thread id, or any unique prefix of one")
    thread.add_argument(
        "--full", action="store_true", help="tool arguments and results in full"
    )
    thread.add_argument(
        "--no-tools", action="store_true", help="hide tool calls and results"
    )
    thread.add_argument("--json", action="store_true", help="everything as JSON")

    return parser


def _load_settings() -> Settings:
    try:
        return get_settings()
    except ValueError as exc:
        raise ScriptError(f"settings: {exc}") from exc


def _db_path(arg: str | None, settings: Settings) -> Path:
    if arg:
        return Path(arg)
    if settings.checkpointer == "memory":
        raise ScriptError(
            "HARNESS_CHECKPOINTER=memory keeps conversations inside the API process; "
            "there is nothing on disk to read (pass --db PATH)"
        )
    if settings.checkpointer != "sqlite":
        raise ScriptError(
            f"HARNESS_CHECKPOINTER={settings.checkpointer}: db.py reads SQLite only "
            "for now (pass --db PATH)"
        )
    return Path(settings.checkpointer_url)


Command = Callable[[Store, argparse.Namespace, Settings], None]

_COMMANDS: dict[str, Command] = {
    "threads": cmd_threads,
    "thread": cmd_thread,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = _load_settings()
        store = Store(_db_path(args.db, settings))
        try:
            _COMMANDS[args.command](store, args, settings)
        finally:
            store.close()
    except ScriptError as exc:
        print(f"db.py: {exc}", file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print(f"db.py: can't read the database: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
