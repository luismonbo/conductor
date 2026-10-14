"""The website agent end to end: it reads its wiki, and browsers see tool names only."""
from __future__ import annotations

import httpx2
import pytest

import harness.api.main as _main
import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, Role, ToolCall
from tests.integration.conftest import _TEST_API_KEY, sse_frames

_READ_INDEX = LLMResponse(
    text="", tool_calls=(ToolCall(id="c1", name="read_wiki_index", arguments={}),)
)
_READ_PAGE = LLMResponse(
    text="",
    tool_calls=(ToolCall(id="c2", name="read_wiki_page", arguments={"page": "projects/harness"}),),
)
_ANSWER = LLMResponse(text="Luis built a production agent harness (Agent Harness).")


def _write_wiki(root):
    (root / "projects").mkdir()
    (root / "index.md").write_text("- [[projects/harness]] — the agent harness\n")
    (root / "projects" / "harness.md").write_text(
        "---\ntitle: Agent Harness\n---\nLuis built a production agent harness.\n"
    )
    return root


@pytest.fixture
async def website_app(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_AGENT", "website")
    monkeypatch.setenv("HARNESS_SHOW_TOOL_DETAILS", "false")
    monkeypatch.setenv("HARNESS_WIKI_DIR", str(_write_wiki(tmp_path)))
    fake = FakeLLMClient([_READ_INDEX, _READ_PAGE, _ANSWER])
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)
    transport = httpx2.ASGITransport(app=_main.app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
    ) as client:
        yield client, fake
    if _main._run_store is not None:
        await _main._run_store._conn.close()


async def test_answers_from_the_wiki_and_shows_browsers_tool_names_only(website_app):
    client, fake = website_app

    response = await client.post("/chat/stream", json={"message": "What did Luis build?"})

    frames = sse_frames(response)
    calls = [f for f in frames if f.get("type") == "tool_call"]
    results = [f for f in frames if f.get("type") == "tool_result"]
    assert [(c["name"], c["args"]) for c in calls] == [("read_wiki_index", {}), ("read_wiki_page", {})]
    assert [r["text"] for r in results] == ["", ""]
    assert frames[-1]["type"] == "final"
    page_result = [m for m in fake.calls[2] if m.role == Role.TOOL][-1]
    assert page_result.content == "# Agent Harness\n\nLuis built a production agent harness."


async def test_the_app_does_not_start_without_the_wiki(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_AGENT", "website")
    with pytest.raises(RuntimeError, match="needs its wiki"):
        async with _main.lifespan(_main.app):
            pass
