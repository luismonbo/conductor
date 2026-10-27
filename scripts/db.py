"""Read-only views of the harness's conversations and run history.

Reads the database the API writes: LangGraph's checkpoints (the conversations),
`runs` (one row per agent run: agent, tokens, stop reason) and `guard_decisions`
(one row per screened message). With HARNESS_CHECKPOINTER=sqlite that's the
harness.sqlite file, opened read-only, so a wrong path is an error instead of a
new empty database. With HARNESS_CHECKPOINTER=postgres it's the Postgres
HARNESS_CHECKPOINTER_URL names, read in read-only transactions; a query against
Neon wakes it for a few minutes. Nothing here writes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg

from harness.config.settings import Settings, get_settings

from _common import (
    ScriptError,
    cut,
    first_line,
    load_settings,
    positive_int,
    print_json,
    print_table,
    resolve,
    visible,
)
from _stores import Conversation, PostgresStore, SqliteStore, Store

_EXAMPLES = """examples:
  uv run scripts/db.py threads
  uv run scripts/db.py thread 10fa
  uv run scripts/db.py thread 10fa --markdown > thread.md
  uv run scripts/db.py usage --days 14
  uv run scripts/db.py guard --blocked
"""

_TITLE_CHARS = 60
_CALL_CHARS = 150
_RESULT_CHARS = 120
_LABELS = {"user": "You", "assistant": "Assistant", "system": "System"}


# --- formatting --------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _short(value: str | None) -> str:
    """A UUID (thread, session or run id) as its first 8 characters."""
    return visible(value[:8]) if value else "-"


def _fmt_time(iso: str | None) -> str:
    if not iso:
        return "-"
    moment = datetime.fromisoformat(iso).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M")


def _count(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


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
    text = visible(_call_text(call))
    return "  → " + (text if full else cut(text, _CALL_CHARS))


def _result_lines(content: str, full: bool) -> list[str]:
    shown = visible(content)
    if full:
        first, *rest = shown.splitlines() or [""]
        return [f"  ← {first}", *(f"    {line}" for line in rest)]
    line = " ".join(shown.split())
    if len(line) <= _RESULT_CHARS:
        return [f"  ← {line}"]
    return [f"  ← {line[:_RESULT_CHARS]} … ({len(content):,} chars)"]


def _speaker_lines(label: str, text: str) -> list[str]:
    first, *rest = visible(text).splitlines() or [""]
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
        print_json(found)
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
                cut(info["title"], _TITLE_CHARS),
            ]
        )
    print_table(
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


def _code(text: str) -> str:
    """Inline Markdown code that survives backticks inside it."""
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _markdown(
    info: dict[str, Any],
    messages: list[Any],
    guard: list[dict[str, Any]],
    *,
    tools: bool,
) -> str:
    """The conversation as Markdown, to export it."""
    tokens = (info["input_tokens"] or 0) + (info["output_tokens"] or 0)
    meta = " · ".join(
        [
            f"Thread {_code(visible(info['thread_id']))}",
            info["agent"] or "agent unknown",
            f"{info['turns'] or 0} turns",
            f"{tokens:,} tokens",
            f"last activity {_fmt_time(info['last_activity'])} UTC",
        ]
    )
    title = cut(info["title"], _TITLE_CHARS) or info["thread_id"]
    lines = [f"# {title}", "", meta, ""]
    for message in messages:
        role = _role(message)
        content = _content(message)
        if role == "tool":
            if tools:
                lines += [
                    "<details>",
                    f"<summary>Tool result ({len(content):,} chars)</summary>",
                    "",
                    visible(content),
                    "",
                    "</details>",
                    "",
                ]
            continue
        if content:
            label = _LABELS.get(role, role.capitalize())
            lines += [f"**{label}:** {visible(content)}", ""]
        if tools:
            for call in _tool_calls(message):
                lines += [f"_Tool call: {_code(visible(_call_text(call)))}_", ""]
    blocked = [row for row in guard if not row["allowed"]]
    if blocked:
        lines += ["## Blocked messages", ""]
        for row in blocked:
            text = " ".join(visible(row["message_text"]).split())
            when = _fmt_time(row["created_at"])
            lines.append(f"- {when} UTC, {row['reason'] or row['stage']}: {text}")
        lines.append("")
    return "\n".join(lines)


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
    print(f"Thread {visible(info['thread_id'])}")
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
        print_table(
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
        print_table(
            ["time (UTC)", "verdict", "stage", "reason", "message"],
            [
                [
                    _fmt_time(row["created_at"]),
                    _verdict(row),
                    row["stage"],
                    row["reason"] or "-",
                    cut(row["message_text"], _TITLE_CHARS),
                ]
                for row in guard
            ],
        )


def cmd_thread(store: Store, args: argparse.Namespace, settings: Settings) -> None:
    candidates = store.thread_ids() + store.run_thread_ids()
    thread_id = resolve(args.thread_id, candidates, "thread")
    conversation = store.conversation(thread_id)
    runs = store.runs(thread_id)
    info = _thread_info(thread_id, conversation, _summarize_runs(runs).get(thread_id))
    messages = conversation.messages if conversation else []
    guard = list(reversed(store.guard_decisions(thread_id=thread_id)))  # oldest first
    if args.json:
        print_json(
            {
                **info,
                "messages": [_message_json(message) for message in messages],
                "runs": runs,
                "guard_decisions": [_guard_json(row) for row in guard],
            }
        )
        return
    if args.markdown:
        print(_markdown(info, messages, guard, tools=not args.no_tools), end="")
        return
    _print_thread(info, messages, runs, guard, full=args.full, tools=not args.no_tools)


def cmd_usage(store: Store, args: argparse.Namespace, settings: Settings) -> None:
    """Counted as the API counts its daily budget: runs started since UTC midnight."""
    today = _utcnow().date()
    days = [(today - timedelta(days=n)).isoformat() for n in range(args.days - 1, -1, -1)]
    runs = store.runs_per_day(days[0])
    blocked = store.blocked_per_day(days[0])
    empty = {"runs": 0, "threads": 0, "tokens": 0}
    per_day = [
        {"day": day, **runs.get(day, empty), "blocked": blocked.get(day, 0)}
        for day in days
    ]
    today_tokens = per_day[-1]["tokens"]
    budget = settings.daily_token_budget
    total_runs, total_tokens = store.run_totals()
    if args.json:
        print_json(
            {
                "days": per_day,
                "today_tokens": today_tokens,
                "daily_token_budget": budget,
                "all_time": {"runs": total_runs, "tokens": total_tokens},
            }
        )
        return
    print_table(
        ["day (UTC)", "runs", "threads", "tokens", "blocked"],
        [
            [
                day["day"],
                str(day["runs"]),
                str(day["threads"]),
                f"{day['tokens']:,}",
                str(day["blocked"]),
            ]
            for day in per_day
        ],
        numeric=frozenset({1, 2, 3, 4}),
    )
    print()
    if budget:
        share = round(100 * today_tokens / budget)
        print(f"today: {today_tokens:,} of {budget:,} daily budget ({share}%)")
    else:
        print(f"today: {today_tokens:,} tokens (daily budget off)")
    print(f"all time: {total_runs:,} runs, {total_tokens:,} tokens")


def cmd_guard(store: Store, args: argparse.Namespace, settings: Settings) -> None:
    rows = store.guard_decisions(blocked_only=args.blocked, limit=args.limit)
    if args.json:
        print_json([_guard_json(row) for row in rows])
        return
    if not rows:
        print("No guard decisions.")
        return
    print_table(
        ["time (UTC)", "verdict", "stage", "reason", "latency", "thread", "message"],
        [
            [
                _fmt_time(row["created_at"]),
                _verdict(row),
                row["stage"],
                row["reason"] or "-",
                "-" if row["latency_ms"] is None else f"{row['latency_ms']} ms",
                _short(row["thread_id"]),
                cut(row["message_text"], _TITLE_CHARS),
            ]
            for row in rows
        ],
    )


# --- command line ------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db", help="a SQLite file to read instead of the API's own database"
    )
    parser = argparse.ArgumentParser(
        prog="db.py",
        description="Conversations, runs and guard decisions in the API's database (read-only).",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    threads = commands.add_parser(
        "threads", parents=[common], help="conversations, newest first"
    )
    threads.add_argument("--limit", type=positive_int, default=20)
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
    output = thread.add_mutually_exclusive_group()
    output.add_argument(
        "--markdown", action="store_true", help="the conversation as Markdown, to export it"
    )
    output.add_argument("--json", action="store_true", help="everything as JSON")

    usage = commands.add_parser(
        "usage", parents=[common], help="tokens per UTC day against the daily budget"
    )
    usage.add_argument(
        "--days", type=positive_int, default=7, help="days back, today included"
    )
    usage.add_argument("--json", action="store_true", help="full rows as JSON")

    guard = commands.add_parser(
        "guard", parents=[common], help="input-guard decisions, newest first"
    )
    guard.add_argument("--limit", type=positive_int, default=20)
    guard.add_argument("--blocked", action="store_true", help="blocked messages only")
    guard.add_argument(
        "--json", action="store_true", help="full rows, with the scores, as JSON"
    )

    return parser


def _open_store(db: str | None, settings: Settings) -> Store:
    """The SQLite file --db names, or else the database the API uses (HARNESS_CHECKPOINTER)."""
    if db:
        return SqliteStore(Path(db))
    if settings.checkpointer == "memory":
        raise ScriptError(
            "HARNESS_CHECKPOINTER=memory keeps conversations inside the API process; "
            "there is nothing on disk to read (pass --db PATH)"
        )
    if settings.checkpointer == "postgres":
        return PostgresStore(settings.checkpointer_url)
    if settings.checkpointer != "sqlite":
        raise ScriptError(
            f"HARNESS_CHECKPOINTER={settings.checkpointer!r} is not memory, sqlite or postgres"
        )
    if "://" in settings.checkpointer_url:
        # A SQLite file path never holds "://", and any URL may carry a password, in
        # whatever scheme. SqliteStore would report it as a missing file, password and all.
        raise ScriptError(
            "HARNESS_CHECKPOINTER_URL is a connection string, not a SQLite file, but "
            "HARNESS_CHECKPOINTER is 'sqlite': for Postgres, set HARNESS_CHECKPOINTER=postgres"
        )
    return SqliteStore(Path(settings.checkpointer_url))


Command = Callable[[Store, argparse.Namespace, Settings], None]

_COMMANDS: dict[str, Command] = {
    "threads": cmd_threads,
    "thread": cmd_thread,
    "usage": cmd_usage,
    "guard": cmd_guard,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(get_settings)
        store = _open_store(args.db, settings)
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
    except psycopg.Error as exc:
        print(f"db.py: can't read the database: {first_line(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
