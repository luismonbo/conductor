"""Integration test fixtures — clear module-level app state between tests."""
from __future__ import annotations

import json

import httpx
import pytest

import harness.api.main as _main
import harness.orchestration.build as build_module
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse


@pytest.fixture(autouse=True)
async def clean_app_state(monkeypatch):
    """Reset all module-level state and force memory checkpointer before each test."""
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "memory")
    _main._running.clear()
    _main._short_term._store.clear()
    _main._registry = None
    _main._run_store = None
    _main._run_store_lock = None
    yield
    for task in list(_main._running.values()):
        if not task.done():
            task.cancel()
    _main._running.clear()
    _main._short_term._store.clear()
    _main._registry = None
    _main._run_store = None
    _main._run_store_lock = None


@pytest.fixture
async def client_with_fake(monkeypatch, tmp_path):
    """App over ASGI with a scripted FakeLLMClient and tmp sqlite storage.

    Overrides clean_app_state's memory checkpointer: the sqlite checkpointer
    and the run store share the tmp file, so /threads has data to serve.
    """
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "h.sqlite"))

    fake = FakeLLMClient([LLMResponse(text="Hello from fake!")])
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)

    transport = httpx.ASGITransport(app=_main.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client, fake

    if _main._run_store is not None:
        await _main._run_store._conn.close()


def sse_frames(resp: httpx.Response) -> list[dict]:
    """All `data:` frames of a fully-buffered SSE response, parsed."""
    return [
        json.loads(line[len("data: "):])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]


def thread_id_from(resp: httpx.Response) -> str:
    return sse_frames(resp)[0]["thread_id"]
