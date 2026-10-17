"""scripts/db.py against databases the harness's own writers produced.

The conversations come from the real loop (fake LLM, calculator tool) and the
runs and guard decisions from the run store, all through open_storage: the rows
`make api` leaves in harness.sqlite, or in Postgres. Every test that takes
`settings` runs against both readers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

import _stores
import scripts.db as db
from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.tools.calculator import CalculatorTool
from harness.config.settings import Settings
from harness.core.guard import GuardDecision
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role, ToolCall
from harness.observability.run_store import RunStore
from harness.observability.token_accumulator import TokenAccumulator
from harness.orchestration.loop import build_graph
from harness.orchestration.storage import open_storage
from tests.conftest import postgres_schema

CHAT = "aaaa1111-0000-4000-8000-000000000001"  # two turns, tool calls, a repeated question
QUIET = "aaaa2222-0000-4000-8000-000000000002"  # one turn and no runs, like older local data
RUNS_ONLY = "cccc3333-0000-4000-8000-000000000003"  # runs, but no stored conversation
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
POEM = "write me a poem\nabout snakes 🐍"  # a blocked first message: no thread

# The run store stamps rows with the current time; the tests pin them to known days.
_RUN_TIMES = {
    "run-1": ("2026-10-14T09:00:00+00:00", "2026-10-14T09:00:06+00:00"),
    "run-2": ("2026-10-15T10:00:00+00:00", "2026-10-15T10:00:04+00:00"),
    "run-3": ("2026-10-15T11:00:00+00:00", "2026-10-15T11:00:02+00:00"),
    "run-4": ("2026-10-15T11:30:00+00:00", None),  # still running
}
_GUARD_TIMES = {
    POEM: "2026-10-13T08:00:00+00:00",
    "What is 2+2?": "2026-10-14T08:59:00+00:00",
    "one more question": "2026-10-15T10:05:00+00:00",
}


def _calculator_call(call_id: str) -> LLMResponse:
    call = ToolCall(id=call_id, name="calculator", arguments={"expression": "2+2"})
    return LLMResponse(text="", tool_calls=(call,))


async def _converse(
    checkpointer,
    thread_id: str,
    questions: list[str],
    replies: list[LLMResponse],
) -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    graph = build_graph(
        llm=FakeLLMClient(replies),
        checkpointer=checkpointer,
        registry=registry,
        system_prompt="You are a test agent.",
    )
    for question in questions:
        config = {"configurable": {"thread_id": thread_id, "event_queue": asyncio.Queue()}}
        state = {
            "messages": [Message(Role.USER, question)],
            "iteration": 0,
            "max_iterations": 8,
        }
        await graph.ainvoke(state, config)


async def _record_runs(store: RunStore) -> None:
    finished = {
        "run-1": (CHAT, "default", 100, 20),
        "run-2": (CHAT, "default", 300, 40),
        "run-3": (RUNS_ONLY, "website", 50, 5),
    }
    for run_id, (thread_id, agent, input_tokens, output_tokens) in finished.items():
        await store.start_run(run_id, thread_id, agent, "fake", "session-1")
        tokens = TokenAccumulator(
            input_tokens=input_tokens, output_tokens=output_tokens, iterations=2
        )
        await store.finish_run(run_id, tokens, "final_answer")
    await store.start_run("run-4", RUNS_ONLY, "website", "fake", "session-1")
    decisions = [
        (
            GuardDecision(
                allowed=False,
                stage="classifier",
                reason="off_topic",
                probabilities={"off_topic": 0.93},
                latency_ms=210,
            ),
            None,
            None,
            POEM,
        ),
        (
            GuardDecision(
                allowed=True,
                stage="classifier",
                probabilities={"off_topic": 0.02},
                latency_ms=190,
            ),
            CHAT,
            "run-1",
            "What is 2+2?",
        ),
        (
            GuardDecision(allowed=False, stage="precap", reason="max_thread_turns"),
            CHAT,
            None,
            "one more question",
        ),
    ]
    for decision, thread_id, run_id, message in decisions:
        await store.record_guard_decision(
            decision, session_id="session-1", thread_id=thread_id, run_id=run_id,
            message=message,
        )


def _storage_settings(checkpointer: str, url: str) -> Settings:
    """What the API runs with, so the test data goes through its own writers."""
    return Settings(
        _env_file=None, auth_enabled=False, input_guard_enabled=False,
        checkpointer=checkpointer, checkpointer_url=url,
    )


async def _build(settings: Settings) -> None:
    storage = await open_storage(settings)
    try:
        await _converse(
            storage.checkpointer,
            CHAT,
            ["What is 2+2?", "What is 2+2?"],
            [
                _calculator_call("c1"),
                LLMResponse(text="It is 4."),
                _calculator_call("c2"),
                LLMResponse(text="Still 4."),
            ],
        )
        await _converse(storage.checkpointer, QUIET, ["Hello"], [LLMResponse(text="Hi there.")])
        await _record_runs(storage.run_store)
    finally:
        await storage.aclose()


async def _create_tables(settings: Settings) -> None:
    """The API's tables with no rows: what its first start leaves in a new database."""
    storage = await open_storage(settings)
    await storage.aclose()


def _set_times_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    with conn:
        for run_id, (started, finished) in _RUN_TIMES.items():
            conn.execute(
                "UPDATE runs SET started_at = ?, finished_at = ? WHERE run_id = ?",
                (started, finished, run_id),
            )
        for message, created in _GUARD_TIMES.items():
            conn.execute(
                "UPDATE guard_decisions SET created_at = ? WHERE message_text = ?",
                (created, message),
            )
    conn.close()


def _when(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def _set_times_postgres(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        for run_id, (started, finished) in _RUN_TIMES.items():
            conn.execute(
                "UPDATE runs SET started_at = %s, finished_at = %s WHERE run_id = %s",
                (_when(started), _when(finished), run_id),
            )
        for message, created in _GUARD_TIMES.items():
            conn.execute(
                "UPDATE guard_decisions SET created_at = %s WHERE message_text = %s",
                (_when(created), message),
            )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """A SQLite database, as `make api` leaves harness.sqlite."""
    path = tmp_path / "harness.sqlite"
    asyncio.run(_build(_storage_settings("sqlite", str(path))))
    _set_times_sqlite(path)
    return path


def _use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "checkpointer": "sqlite",
        "daily_token_budget": 790,
        **overrides,
    }
    settings = Settings(
        _env_file=None, auth_enabled=False, input_guard_enabled=False, **values
    )
    monkeypatch.setattr(db, "get_settings", lambda: settings)
    monkeypatch.setattr(db, "_utcnow", lambda: NOW)
    return settings


@pytest.fixture(params=["sqlite", "postgres"])
def settings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The API's settings, pointing at the test data in SQLite or in Postgres."""
    if request.param == "sqlite":
        url = str(request.getfixturevalue("database"))
    else:
        url = request.getfixturevalue("postgres_dsn")  # skips without Postgres
        asyncio.run(_build(_storage_settings("postgres", url)))
        _set_times_postgres(url)
    return _use_settings(monkeypatch, checkpointer=request.param, checkpointer_url=url)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = db.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# --- threads -----------------------------------------------------------------


def test_threads_lists_conversations_newest_first(settings, capsys):
    code, out, _ = run(capsys, "threads", "--json")
    assert code == 0
    threads = json.loads(out)
    assert [t["thread_id"] for t in threads] == [QUIET, CHAT]
    chat = threads[1]
    assert (chat["agent"], chat["turns"]) == ("default", 2)
    assert (chat["input_tokens"], chat["output_tokens"]) == (400, 60)
    assert chat["title"] == "What is 2+2?"
    assert chat["messages"] == 8
    assert chat["last_activity"]


def test_a_thread_without_runs_shows_dashes(settings, capsys):
    code, out, _ = run(capsys, "threads")
    assert code == 0
    quiet = next(line for line in out.splitlines() if line.startswith("aaaa2222"))
    assert quiet.endswith("Hello")
    assert quiet.split()[3:6] == ["-", "-", "-"]  # agent, turns, tokens


def test_threads_filters_by_agent(settings, capsys):
    _, out, _ = run(capsys, "threads", "--agent", "default", "--json")
    assert [t["thread_id"] for t in json.loads(out)] == [CHAT]


def test_threads_respects_the_limit(settings, capsys):
    _, out, _ = run(capsys, "threads", "--limit", "1", "--json")
    assert [t["thread_id"] for t in json.loads(out)] == [QUIET]


# --- opening the database ----------------------------------------------------


def test_a_missing_database_is_an_error_and_creates_nothing(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "nope.sqlite"
    _use_settings(monkeypatch, checkpointer_url=str(missing))
    code, _, err = run(capsys, "threads")
    assert code == 1
    assert "no database at" in err
    assert not missing.exists()


def test_the_db_flag_overrides_the_settings(database, tmp_path, monkeypatch, capsys):
    _use_settings(monkeypatch, checkpointer_url=str(tmp_path / "elsewhere.sqlite"))
    code, out, _ = run(capsys, "threads", "--db", str(database), "--json")
    assert code == 0
    assert len(json.loads(out)) == 2


def test_memory_mode_has_nothing_to_read(monkeypatch, capsys):
    _use_settings(monkeypatch, checkpointer="memory")
    code, _, err = run(capsys, "threads")
    assert code == 1
    assert "nothing on disk" in err


def test_an_unknown_checkpointer_is_an_error(monkeypatch, capsys):
    _use_settings(monkeypatch, checkpointer="redis")
    code, _, err = run(capsys, "threads")
    assert code == 1
    assert "'redis'" in err


@pytest.mark.parametrize(
    "scheme", ["postgresql", "postgres", "postgresql+psycopg", "POSTGRESQL"]
)
def test_a_postgres_url_under_the_sqlite_checkpointer_is_never_printed(
    scheme, monkeypatch, capsys
):
    """A forgotten HARNESS_CHECKPOINTER=postgres must not echo the connection string
    as if it were a missing file, whatever its scheme: any URL may carry a password."""
    _use_settings(
        monkeypatch,
        checkpointer="sqlite",
        checkpointer_url=f"{scheme}://harness:hunter2@db.example.com/harness",
    )
    code, out, err = run(capsys, "threads")
    assert code == 1
    assert err == (
        "db.py: HARNESS_CHECKPOINTER_URL is a connection string, not a SQLite file, but "
        "HARNESS_CHECKPOINTER is 'sqlite': for Postgres, set HARNESS_CHECKPOINTER=postgres\n"
    )
    assert "hunter2" not in out + err


def test_an_unreachable_postgres_is_one_line_without_the_password(monkeypatch, capsys):
    _use_settings(
        monkeypatch,
        checkpointer="postgres",
        checkpointer_url="postgresql://harness:hunter2@127.0.0.1:1/harness",
    )
    code, _, err = run(capsys, "threads")
    assert code == 1
    assert "can't reach Postgres at harness@127.0.0.1:1/harness" in err
    assert "make infra" in err
    assert "hunter2" not in err
    assert err.count("\n") == 1


def test_a_postgres_database_the_api_never_opened(postgres_dsn, monkeypatch, capsys):
    """A new Neon database, before the first deploy creates the tables."""
    _use_settings(monkeypatch, checkpointer="postgres", checkpointer_url=postgres_dsn)
    code, out, _ = run(capsys, "threads")
    assert (code, out) == (0, "No threads.\n")
    code, out, _ = run(capsys, "usage", "--json")
    assert code == 0
    assert json.loads(out)["all_time"] == {"runs": 0, "tokens": 0}


def test_a_failed_postgres_setup_closes_its_connection(postgres_dsn, monkeypatch):
    """A store that fails half-built never reaches main()'s finally: nothing else
    can close its connection."""
    asyncio.run(_create_tables(_storage_settings("postgres", postgres_dsn)))
    connections: list[psycopg.Connection] = []
    connect = _stores.connect_postgres

    def spy(dsn: str) -> psycopg.Connection:
        connections.append(connect(dsn))
        return connections[-1]

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("saver refused")

    monkeypatch.setattr(_stores, "connect_postgres", spy)
    monkeypatch.setattr(_stores, "PostgresSaver", refuse)
    with pytest.raises(RuntimeError, match="saver refused"):
        db.PostgresStore(postgres_dsn)
    assert [conn.closed for conn in connections] == [True]


# --- thread ------------------------------------------------------------------


def test_thread_shows_every_message(settings, capsys):
    code, out, _ = run(capsys, "thread", "aaaa1")
    assert code == 0
    assert out.count("You: What is 2+2?") == 2
    assert out.count('→ calculator(expression="2+2")') == 2
    assert out.count("← 4") == 2
    assert "Assistant: It is 4." in out
    assert "Assistant: Still 4." in out


def test_thread_json_carries_messages_runs_and_guard(settings, capsys):
    _, out, _ = run(capsys, "thread", CHAT, "--json")
    thread = json.loads(out)
    roles = [m["role"] for m in thread["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"] * 2
    assert thread["messages"][1]["tool_calls"] == [
        {"id": "c1", "name": "calculator", "arguments": {"expression": "2+2"}}
    ]
    assert [r["run_id"] for r in thread["runs"]] == ["run-1", "run-2"]
    assert [(g["message_text"], g["allowed"]) for g in thread["guard_decisions"]] == [
        ("What is 2+2?", True),
        ("one more question", False),
    ]


def test_thread_lists_its_runs_and_the_blocked_follow_up(settings, capsys):
    _, out, _ = run(capsys, "thread", "aaaa1")
    runs = out.split("\nRuns\n")[1]
    assert "final_answer" in runs and "6.0s" in runs
    guard = out.split("\nGuard\n")[1]
    assert "blocked" in guard and "one more question" in guard


def test_no_tools_hides_tool_traffic(settings, capsys):
    _, out, _ = run(capsys, "thread", "aaaa1", "--no-tools")
    assert "→" not in out and "←" not in out
    assert "Assistant: Still 4." in out


def test_long_tool_results_are_cut_unless_full():
    [cut] = db._result_lines("word " * 100, full=False)
    assert cut.endswith("… (500 chars)")
    assert len(cut) < 140
    assert db._result_lines("line one\nline two", full=True) == [
        "  ← line one",
        "    line two",
    ]


def test_long_tool_arguments_are_cut_unless_full():
    call = ToolCall(id="c", name="search_documents", arguments={"query": "x" * 300})
    assert len(db._call_line(call, full=False)) <= 4 + 150
    assert db._call_line(call, full=True).endswith('"' + "x" * 300 + '")')


def test_an_ambiguous_prefix_lists_the_matches(settings, capsys):
    code, _, err = run(capsys, "thread", "aaaa")
    assert code == 1
    assert CHAT in err and QUIET in err


def test_an_unknown_thread_is_an_error(settings, capsys):
    code, _, err = run(capsys, "thread", "ffff")
    assert code == 1
    assert "no thread matching 'ffff'" in err


def test_a_thread_with_runs_only(settings, capsys):
    code, out, _ = run(capsys, "thread", "cccc")
    assert code == 0
    assert "(no conversation stored)" in out
    assert "final_answer" in out


def test_a_database_from_before_visitor_sessions(tmp_path, monkeypatch, capsys):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, thread_id TEXT, agent TEXT, "
            "backend TEXT, started_at TEXT, finished_at TEXT, stopped_reason TEXT, "
            "input_tokens INTEGER, output_tokens INTEGER, iterations INTEGER)"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('r1', 'legacy-1', 'default', 'fake', "
            "'2026-08-01T10:00:00+00:00', '2026-08-01T10:00:03+00:00', "
            "'final_answer', 10, 2, 1)"
        )
    conn.close()
    _use_settings(monkeypatch, checkpointer_url=str(path))
    code, out, err = run(capsys, "thread", "legacy")
    assert code == 0, err
    assert "session -" in out
    assert "(no conversation stored)" in out


def test_a_message_in_an_unexpected_shape_prints_as_text():
    lines = db._transcript_lines([{"role": "x"}, "plain text"], full=False, tools=True)
    assert lines == ["Unknown: {'role': 'x'}", "Unknown: plain text"]


# --- markdown export ---------------------------------------------------------


def test_markdown_export(settings, capsys):
    code, out, _ = run(capsys, "thread", "aaaa1", "--markdown")
    assert code == 0
    assert out.startswith("# What is 2+2?\n")
    assert out.count("**You:** What is 2+2?") == 2
    assert "**Assistant:** Still 4." in out
    assert '_Tool call: `calculator(expression="2+2")`_' in out
    assert "<summary>Tool result (1 chars)</summary>" in out
    assert "## Blocked messages" in out and "one more question" in out


def test_markdown_without_tools(settings, capsys):
    _, out, _ = run(capsys, "thread", "aaaa1", "--markdown", "--no-tools")
    assert "Tool call" not in out and "<details>" not in out
    assert "**Assistant:** It is 4." in out


def test_markdown_and_json_are_exclusive(settings, capsys):
    with pytest.raises(SystemExit) as exit_info:
        db.main(["thread", "aaaa1", "--markdown", "--json"])
    assert exit_info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_inline_code_survives_backticks():
    assert db._code("plain") == "`plain`"
    assert db._code("has ` tick") == "`` has ` tick ``"


# --- usage -------------------------------------------------------------------


def test_usage_per_day_against_the_budget(settings, capsys):
    code, out, _ = run(capsys, "usage", "--days", "3", "--json")
    assert code == 0
    usage = json.loads(out)
    assert usage["days"] == [
        {"day": "2026-10-13", "runs": 0, "threads": 0, "tokens": 0, "blocked": 1},
        {"day": "2026-10-14", "runs": 1, "threads": 1, "tokens": 120, "blocked": 0},
        {"day": "2026-10-15", "runs": 3, "threads": 2, "tokens": 395, "blocked": 1},
    ]
    assert usage["today_tokens"] == 395
    assert usage["all_time"] == {"runs": 4, "tokens": 515}


def test_usage_text_shows_the_budget_line(settings, capsys):
    _, out, _ = run(capsys, "usage", "--days", "3")
    assert "today: 395 of 790 daily budget (50%)" in out
    assert "all time: 4 runs, 515 tokens" in out


def test_usage_with_the_budget_off(settings, monkeypatch, capsys):
    _use_settings(
        monkeypatch,
        checkpointer=settings.checkpointer,
        checkpointer_url=settings.checkpointer_url,
        daily_token_budget=0,
    )
    _, out, _ = run(capsys, "usage")
    assert "today: 395 tokens (daily budget off)" in out


# --- guard -------------------------------------------------------------------


def test_guard_blocked_newest_first(settings, capsys):
    code, out, _ = run(capsys, "guard", "--blocked", "--json")
    assert code == 0
    rows = json.loads(out)
    assert [r["message_text"] for r in rows] == ["one more question", POEM]
    assert rows[1]["thread_id"] is None
    assert rows[1]["probabilities"] == {"off_topic": 0.93}


def test_guard_text_marks_a_blocked_first_message(settings, capsys):
    _, out, _ = run(capsys, "guard")
    poem = [line for line in out.splitlines() if "write me a poem" in line]
    assert len(poem) == 1  # the newline inside the message stays inside one row
    assert "about snakes 🐍" in poem[0]
    assert "blocked" in poem[0] and "off_topic" in poem[0] and "210 ms" in poem[0]
    assert " - " in poem[0]  # no thread


# --- read-only ---------------------------------------------------------------


def _snapshot(settings: Settings) -> object:
    """Every byte of the SQLite file, or every row of the Postgres schema."""
    if settings.checkpointer == "sqlite":
        return hashlib.sha256(Path(settings.checkpointer_url).read_bytes()).hexdigest()
    with psycopg.connect(settings.checkpointer_url) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() ORDER BY 1"
            )
        ]
        return {
            table: conn.execute(
                sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(table))
            ).fetchall()
            for table in tables
        }


def test_every_command_leaves_the_database_untouched(settings, capsys):
    before = _snapshot(settings)
    for argv in (
        ["threads"],
        ["thread", "aaaa1"],
        ["thread", "aaaa1", "--markdown"],
        ["thread", "cccc"],
        ["usage"],
        ["guard"],
    ):
        assert db.main(argv) == 0
    capsys.readouterr()
    assert _snapshot(settings) == before


def test_the_postgres_reader_refuses_writes(postgres_dsn):
    """No command writes, so the test above can't tell a session left writable."""
    store = db.PostgresStore(postgres_dsn)
    try:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            store._conn.execute("CREATE TABLE should_not_exist (a int)")
    finally:
        store.close()


# --- session time zone -------------------------------------------------------

# In New York (UTC-4 in October) both fall on the 14th: a day bucket or a timestamp
# that follows the session's zone instead of UTC lands on the wrong day.
_LATE_RUN = datetime(2026, 10, 15, 2, 0, tzinfo=timezone.utc)  # 22:00 on the 14th there
_LATE_BLOCK = datetime(2026, 10, 15, 3, 30, tzinfo=timezone.utc)  # 23:30 the 14th there


def _insert_late_night_rows(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO runs "
            "(run_id, thread_id, agent, backend, started_at, session_id) "
            "VALUES ('run-late', %s, 'default', 'fake', %s, 'session-1')",
            (RUNS_ONLY, _LATE_RUN),
        )
        conn.execute(
            "INSERT INTO guard_decisions (decision_id, created_at, session_id, "
            "thread_id, allowed, reason, stage, message_chars, message_text) "
            "VALUES ('decision-late', %s, 'session-1', %s, false, 'off_topic', "
            "'classifier', 4, 'late')",
            (_LATE_BLOCK, RUNS_ONLY),
        )


def test_postgres_reads_are_utc_whatever_the_session_time_zone(monkeypatch, capsys):
    """The session's zone must not move a row to another day or change how its time
    reads: both stay the UTC ones the SQLite reader gives."""
    with postgres_schema("-c TimeZone=America/New_York") as dsn:
        asyncio.run(_create_tables(_storage_settings("postgres", dsn)))
        _insert_late_night_rows(dsn)
        _use_settings(monkeypatch, checkpointer="postgres", checkpointer_url=dsn)

        _, out, _ = run(capsys, "usage", "--json")
        per_day = {day["day"]: day for day in json.loads(out)["days"]}
        assert (per_day["2026-10-15"]["runs"], per_day["2026-10-15"]["blocked"]) == (1, 1)
        assert (per_day["2026-10-14"]["runs"], per_day["2026-10-14"]["blocked"]) == (0, 0)

        _, out, _ = run(capsys, "thread", RUNS_ONLY, "--json")
        thread = json.loads(out)
        assert thread["runs"][0]["started_at"] == "2026-10-15T02:00:00+00:00"
        assert thread["guard_decisions"][0]["created_at"] == "2026-10-15T03:30:00+00:00"

        _, out, _ = run(capsys, "thread", RUNS_ONLY)
        assert "2026-10-15 02:00" in out
        assert "2026-10-15 03:30" in out


# --- untrusted text ------------------------------------------------------------

# Visitor messages and the sid cookie are untrusted: raw ESC, BEL, CR, C1 and
# bidi-override characters would rewrite the terminal that prints them.
HOSTILE = "hi \x1b]0;pwned\x07\x1b[2K\x9b31m\u202eevil\u202c\rthere"
HOSTILE_SID = "sid\x1b[2K"
_RAW_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


async def _build_hostile(settings: Settings) -> None:
    storage = await open_storage(settings)
    try:
        await _converse(
            storage.checkpointer,
            CHAT,
            [HOSTILE],
            [_calculator_call("c1"), LLMResponse(text=HOSTILE)],
        )
        await storage.run_store.start_run("run-h", CHAT, "default", "fake", HOSTILE_SID)
        await storage.run_store.record_guard_decision(
            GuardDecision(allowed=False, stage="classifier", reason="off_topic"),
            session_id=HOSTILE_SID, thread_id=CHAT, run_id=None, message=HOSTILE,
        )
    finally:
        await storage.aclose()


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_visitor_control_characters_never_reach_the_terminal(
    backend, request, tmp_path, monkeypatch, capsys
):
    if backend == "sqlite":
        url = str(tmp_path / "hostile.sqlite")
    else:
        url = request.getfixturevalue("postgres_dsn")
    asyncio.run(_build_hostile(_storage_settings(backend, url)))
    _use_settings(monkeypatch, checkpointer=backend, checkpointer_url=url)
    for argv in (
        ["threads"],
        ["threads", "--json"],
        ["thread", CHAT],
        ["thread", CHAT, "--full"],
        ["thread", CHAT, "--markdown"],
        ["thread", CHAT, "--json"],
        ["guard"],
        ["guard", "--json"],
    ):
        code, out, err = run(capsys, *argv)
        assert code == 0, (argv, err)
        assert not _RAW_CONTROLS.search(out), argv
    _, out, _ = run(capsys, "guard")
    assert "\\x1b]0;pwned\\x07" in out  # shown as text, not executed
    _, out, _ = run(capsys, "guard", "--json")
    assert json.loads(out)[0]["message_text"] == HOSTILE  # JSON keeps the exact text


def test_tool_arguments_and_results_are_shown_not_executed():
    call = ToolCall(id="c", name="search_documents", arguments={"query": HOSTILE})
    lines = [
        db._call_line(call, full=False),
        db._call_line(call, full=True),
        *db._result_lines(HOSTILE, full=False),
        *db._result_lines(HOSTILE, full=True),
    ]
    assert not any(_RAW_CONTROLS.search(line) for line in lines)


def _broken_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Invalid settings whose input holds a secret; the real .env is kept out."""
    for key in [key for key in os.environ if key.startswith("HARNESS_")]:
        monkeypatch.delenv(key)
    return Settings(
        _env_file=None, azure_api_key="SECRETkey123", auth_enabled=True, api_key=""
    )


def test_a_settings_error_is_one_line_without_the_input(monkeypatch, capsys):
    monkeypatch.setattr(db, "get_settings", lambda: _broken_settings(monkeypatch))
    code, _, err = run(capsys, "threads")
    assert code == 1
    assert err.startswith("db.py: settings: ") and err.count("\n") == 1
    assert "HARNESS_API_KEY" in err
    assert "SECRET" not in err

