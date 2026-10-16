"""The API on Postgres: what the production target runs, against the test server."""
from __future__ import annotations

import logging

import httpx2
import pytest

import harness.api.main as _main
import harness.orchestration.build as build_module
from harness.adapters.guard.fake import FakeClassifier
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse
from harness.observability.logging_setup import configure_logging
from harness.security.input_guard import GuardLimits, InputGuard
from tests.integration.conftest import _TEST_API_KEY, thread_id_from

_SECRET = "my secret plan is 7731"


@pytest.fixture
async def postgres_client(monkeypatch, postgres_dsn):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "postgres")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", postgres_dsn)
    fake = FakeLLMClient([LLMResponse(text="Hello from fake!")], repeat_last=True)
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)
    transport = httpx2.ASGITransport(app=_main.app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="https://test",
        headers={"Authorization": f"Bearer {_TEST_API_KEY}"},
    ) as client:
        yield client
    # Close the pool before postgres_dsn drops the test's schema.
    if _main._runtime is not None:
        await _main._runtime.aclose()
        _main._runtime = None


async def _restart() -> None:
    """What a new revision does: this runtime closes, the next request opens another."""
    assert _main._runtime is not None
    await _main._runtime.aclose()
    _main._runtime = None


async def test_a_thread_survives_a_restart(postgres_client):
    response = await postgres_client.post("/chat/stream", json={"message": "remember the milk"})
    thread_id = thread_id_from(response)

    await _restart()

    [thread] = (await postgres_client.get("/threads")).json()["threads"]
    assert thread["thread_id"] == thread_id
    assert thread["title"].startswith("remember the milk")
    assert thread["last_at"].endswith("+00:00")
    transcript = (await postgres_client.get(f"/threads/{thread_id}")).json()["messages"]
    assert [m["role"] for m in transcript] == ["user", "assistant"]


async def test_debug_logging_never_carries_the_visitors_message(postgres_client, caplog):
    caplog.set_level(logging.DEBUG)  # an operator's root DEBUG config
    # psycopg pins its own logger (and psycopg.pool's) to WARNING at import, so root
    # DEBUG alone leaves it silent: switch it on, as an operator debugging connections would.
    caplog.set_level(logging.DEBUG, logger="psycopg")
    configure_logging("DEBUG")
    # A classifier decision, so the guard log row (which holds the text) is written too.
    _main._input_guard = InputGuard(
        FakeClassifier({"off_topic": 0.10}), GuardLimits(), timeout_seconds=1.0
    )

    response = await postgres_client.post("/chat/stream", json={"message": _SECRET})

    assert response.status_code == 200
    assert any(record.name == "psycopg" for record in caplog.records)  # DEBUG really ran
    assert _SECRET not in caplog.text
