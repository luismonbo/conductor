"""The API serves one runtime: the entry agent HARNESS_AGENT named at startup."""
from __future__ import annotations

import asyncio

import httpx2
import pytest

import harness.api.main as _main
import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, Message, Role
from harness.orchestration.agents import AgentLoadError
from tests.integration.conftest import _TEST_API_KEY, thread_id_from

GOOD_PROMPT = "You are the good fixture agent."


@pytest.fixture
async def good_agent_app(monkeypatch, tmp_path, fixture_agents):
    """The app with HARNESS_AGENT=good, loaded from tests/fixtures/agents."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "h.sqlite"))
    monkeypatch.setenv("HARNESS_AGENT", "good")
    fake = FakeLLMClient([LLMResponse(text="Hello from the good agent.")], repeat_last=True)
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)
    transport = httpx2.ASGITransport(app=_main.app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
    ) as client:
        yield client, fake


async def test_chat_runs_the_entry_agent(good_agent_app):
    client, fake = good_agent_app
    await client.post("/chat/stream", json={"message": "hi"})
    assert fake.calls[0][0] == Message(role=Role.SYSTEM, content=GOOD_PROMPT)


async def test_a_client_naming_another_agent_still_gets_the_entry_agent(good_agent_app):
    client, fake = good_agent_app
    response = await client.post("/chat/stream", json={"message": "hi", "agent": "default"})
    assert response.status_code == 200
    assert fake.calls[0][0].content == GOOD_PROMPT


async def test_a_thread_reloads_from_the_entry_agent(good_agent_app):
    client, _ = good_agent_app
    first = await client.post("/chat/stream", json={"message": "hi"})
    thread = await client.get(f"/threads/{thread_id_from(first)}")
    assert thread.status_code == 200
    assert [m["role"] for m in thread.json()["messages"]] == ["user", "assistant"]


async def test_requests_keep_the_agent_built_at_startup(good_agent_app, monkeypatch):
    client, fake = good_agent_app
    await client.post("/chat/stream", json={"message": "hi"})
    monkeypatch.setenv("HARNESS_AGENT", "default")  # e.g. .env edited while the app runs
    await client.post("/chat/stream", json={"message": "hi again"})
    assert fake.calls[1][0].content == GOOD_PROMPT


async def test_concurrent_first_requests_build_one_runtime(monkeypatch, fixture_agents):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_AGENT", "good")
    real_build_runtime = _main.build_runtime
    builds = 0

    async def counting_build_runtime(settings):
        nonlocal builds
        builds += 1
        await asyncio.sleep(0)  # give the second caller a chance to race
        return await real_build_runtime(settings)

    monkeypatch.setattr(_main, "build_runtime", counting_build_runtime)
    first, second = await asyncio.gather(_main._get_runtime(), _main._get_runtime())
    assert builds == 1
    assert first is second


async def test_startup_fails_when_the_entry_agent_is_missing(monkeypatch):
    monkeypatch.setenv("HARNESS_AGENT", "no_such_agent")
    with pytest.raises(AgentLoadError, match="no_such_agent"):
        async with _main.lifespan(_main.app):
            pass


async def test_a_failed_startup_closes_the_storage(monkeypatch, tmp_path):
    # The runtime opens its storage before the guard is built. If a later startup
    # step fails, those connections (aiosqlite's worker threads are not daemons)
    # would keep the process alive after uvicorn gives up.
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_EMBEDDING_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_RAG_VECTOR_STORE_BACKEND", "in_memory")
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "h.sqlite"))
    opened = []
    real_open_storage = build_module.open_storage

    async def recording_open_storage(settings):
        opened.append(await real_open_storage(settings))
        return opened[-1]

    def failing_guard(settings):
        raise ValueError("malformed guard setting")

    monkeypatch.setattr(build_module, "open_storage", recording_open_storage)
    monkeypatch.setattr(_main, "build_input_guard", failing_guard)

    with pytest.raises(ValueError, match="malformed guard setting"):
        async with _main.lifespan(_main.app):
            pass

    [storage] = opened
    for conn in (storage.checkpointer.conn, storage.run_store._conn):
        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("select 1")
