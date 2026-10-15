"""scripts/db.py against a database the harness's own writers produced.

The conversations come from the real loop (fake LLM, calculator tool) through
AsyncSqliteSaver; the runs and guard decisions from RunStore. The same bytes
`make api` leaves in harness.sqlite.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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

CHAT = "aaaa1111-0000-4000-8000-000000000001"  # two turns, tool calls, a repeated question
QUIET = "aaaa2222-0000-4000-8000-000000000002"  # one turn and no runs, like older local data
RUNS_ONLY = "cccc3333-0000-4000-8000-000000000003"  # runs, but no stored conversation
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
POEM = "write me a poem\nabout snakes 🐍"  # a blocked first message: no thread

# RunStore stamps rows with the current time; the tests pin them to known days.
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
    conn: aiosqlite.Connection,
    thread_id: str,
    questions: list[str],
    replies: list[LLMResponse],
) -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    graph = build_graph(
        llm=FakeLLMClient(replies),
        checkpointer=AsyncSqliteSaver(conn),
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


async def _record_runs(conn: aiosqlite.Connection) -> None:
    store = RunStore(conn)
    await store.create_table()
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


async def _build(path: Path) -> None:
    async with aiosqlite.connect(path) as conn:
        await _converse(
            conn,
            CHAT,
            ["What is 2+2?", "What is 2+2?"],
            [
                _calculator_call("c1"),
                LLMResponse(text="It is 4."),
                _calculator_call("c2"),
                LLMResponse(text="Still 4."),
            ],
        )
        await _converse(conn, QUIET, ["Hello"], [LLMResponse(text="Hi there.")])
        await _record_runs(conn)


def _set_times(path: Path) -> None:
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


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "harness.sqlite"
    asyncio.run(_build(path))
    _set_times(path)
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


@pytest.fixture
def settings(database: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    return _use_settings(monkeypatch, checkpointer_url=str(database))


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


@pytest.mark.parametrize(
    ("checkpointer", "message"),
    [("memory", "nothing on disk"), ("postgres", "SQLite only")],
)
def test_checkpointers_without_a_file(checkpointer, message, monkeypatch, capsys):
    _use_settings(monkeypatch, checkpointer=checkpointer)
    code, _, err = run(capsys, "threads")
    assert code == 1
    assert message in err


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
