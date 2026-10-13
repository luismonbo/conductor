"""What the API sends a browser (SL-3): tool detail, error text, /health."""
from __future__ import annotations

import logging

import httpx2

import harness.api.main as _main
import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.llm.errors import ContentFiltered
from harness.core.types import LLMResponse, ToolCall
from harness.observability.run_store import RunStore
from harness.security.egress import CONTENT_FILTERED_TEXT, UNAVAILABLE_TEXT
from tests.integration.conftest import _TEST_API_KEY, sse_frames, thread_id_from

_CALCULATE = LLMResponse(
    text="", tool_calls=(ToolCall(id="c1", name="calculator", arguments={"expression": "6*7"}),),
)


def _use_llm(monkeypatch, llm) -> None:
    """Serve `llm` from the next runtime build (the runtime is built on the first request)."""
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: llm)


class _FailingLLM:
    """Answers its first calls with `before` (one LLMResponse per call), then
    streams `tokens` and raises `exc`."""

    def __init__(self, exc: BaseException, before=(), tokens=()) -> None:
        self._exc = exc
        self._before = list(before)
        self._tokens = list(tokens)

    @property
    def model_id(self) -> str:
        return "failing"

    async def generate(self, messages, tools=None, model=None):
        raise NotImplementedError

    async def stream(self, messages, tools=None, model=None):
        if self._before:
            yield self._before.pop(0)
            return
        for token in self._tokens:
            yield token
        raise self._exc


async def test_tool_detail_stays_out_of_the_stream_and_the_transcript(client_with_fake, monkeypatch):
    client, _ = client_with_fake
    monkeypatch.setenv("HARNESS_SHOW_TOOL_DETAILS", "false")
    _use_llm(monkeypatch, FakeLLMClient([_CALCULATE, LLMResponse(text="It is 42.")]))

    response = await client.post("/chat/stream", json={"message": "what is 6*7?"})

    frames = sse_frames(response)
    call = next(f for f in frames if f.get("type") == "tool_call")
    result = next(f for f in frames if f.get("type") == "tool_result")
    assert (call["name"], call["args"]) == ("calculator", {})
    assert (result["name"], result["text"], result["is_error"]) == ("calculator", "", False)
    assert frames[-1]["type"] == "final"
    transcript = (await client.get(f"/threads/{thread_id_from(response)}")).json()["messages"]
    assert transcript[1]["tool_calls"] == [{"name": "calculator", "args": {}, "call_id": "c1"}]
    assert (transcript[2]["role"], transcript[2]["content"]) == ("tool", "")


async def test_tool_detail_reaches_the_browser_where_the_deployment_allows_it(client_with_fake, monkeypatch):
    client, _ = client_with_fake
    monkeypatch.setenv("HARNESS_SHOW_TOOL_DETAILS", "true")
    _use_llm(monkeypatch, FakeLLMClient([_CALCULATE, LLMResponse(text="It is 42.")]))

    response = await client.post("/chat/stream", json={"message": "what is 6*7?"})

    frames = sse_frames(response)
    assert next(f for f in frames if f.get("type") == "tool_call")["args"] == {"expression": "6*7"}
    assert next(f for f in frames if f.get("type") == "tool_result")["text"] == "42"
    transcript = (await client.get(f"/threads/{thread_id_from(response)}")).json()["messages"]
    assert transcript[1]["tool_calls"][0]["args"] == {"expression": "6*7"}
    assert transcript[2]["content"] == "42"


async def test_a_failed_run_sends_the_generic_sentence_not_the_exception(client_with_fake, monkeypatch):
    client, _ = client_with_fake
    _use_llm(monkeypatch, _FailingLLM(ValueError("leaked /app/data/raw/docs/010.md body")))

    response = await client.post("/chat/stream", json={"message": "hi"})

    assert sse_frames(response)[-1] == {
        "type": "error", "text": UNAVAILABLE_TEXT, "name": "", "args": {}, "call_id": "",
        "is_error": False, "stopped_reason": "error",
    }
    assert "leaked" not in response.text


async def test_a_filtered_prompt_mid_run_reaches_the_browser_as_the_filter_sentence(
    client_with_fake, monkeypatch, caplog
):
    # Review Focus 1: a searched document trips the filter on the call after a tool.
    client, _ = client_with_fake
    monkeypatch.setenv("HARNESS_SHOW_TOOL_DETAILS", "false")
    _use_llm(monkeypatch, _FailingLLM(ContentFiltered("prompt", ("violence",)), before=[_CALCULATE]))

    with caplog.at_level(logging.WARNING, logger="harness.api.main"):
        response = await client.post("/chat/stream", json={"message": "what is 6*7?"})

    frames = sse_frames(response)
    assert [f.get("type") for f in frames[1:]] == ["tool_call", "tool_result", "error"]
    assert (frames[-1]["text"], frames[-1]["stopped_reason"]) == (CONTENT_FILTERED_TEXT, "content_filter")
    cursor = await _main._run_store._conn.execute("SELECT stopped_reason FROM runs")
    assert await cursor.fetchall() == [("content_filter",)]
    [record] = [r for r in caplog.records if "content filter" in r.getMessage()]
    assert "6*7" not in record.getMessage()  # never the visitor's text


async def test_a_filtered_answer_is_not_kept_in_the_thread(client_with_fake, monkeypatch):
    # Review Focus 5: the partial answer streamed before the filter cut it off
    # never reaches the checkpoint, so a reload shows the question alone.
    client, _ = client_with_fake
    _use_llm(monkeypatch, FakeLLMClient([
        LLMResponse(text="The first part", finish_reason="content_filter"),
    ]))

    response = await client.post("/chat/stream", json={"message": "tell me a story"})

    frames = sse_frames(response)
    assert [f["type"] for f in frames[1:]] == ["token", "token", "token", "error"]
    assert frames[-1]["stopped_reason"] == "content_filter"
    transcript = (await client.get(f"/threads/{thread_id_from(response)}")).json()["messages"]
    assert [m["role"] for m in transcript] == ["user"]


async def test_a_failure_before_the_stream_starts_is_a_bare_500(monkeypatch):
    # Review Focus 3: nothing about the exception reaches the caller.
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")

    async def _broken_start_run(self, *args, **kwargs):
        raise RuntimeError("database is locked at /app/state/harness.sqlite")

    monkeypatch.setattr(RunStore, "start_run", _broken_start_run)
    transport = httpx2.ASGITransport(app=_main.app, raise_app_exceptions=False)
    async with httpx2.AsyncClient(
        transport=transport, base_url="https://test",
        headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
    ) as client:
        response = await client.post("/chat/stream", json={"message": "hi"})

    assert response.status_code == 500
    assert "harness.sqlite" not in response.text
    assert "database is locked" not in response.text


async def test_health_names_nothing_about_the_deployment(client_with_fake):
    client, _ = client_with_fake

    assert (await client.get("/health")).json() == {"status": "ok"}
