"""A visitor sees only their own threads and cannot read anyone else's."""
import httpx2
import pytest

import harness.api.main as _main
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse
from tests.integration.conftest import _TEST_API_KEY, thread_id_from


@pytest.fixture
async def two_clients(monkeypatch, tmp_path):
    """Two independent cookie jars against one app and one sqlite file."""
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "h.sqlite"))

    import harness.orchestration.build as build_module

    fake = FakeLLMClient([LLMResponse(text="hi")] * 10)
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)

    transport = httpx2.ASGITransport(app=_main.app)
    headers = {"Authorization": f"Bearer {_TEST_API_KEY}"}
    async with httpx2.AsyncClient(transport=transport, base_url="http://test", headers=headers) as a:
        async with httpx2.AsyncClient(transport=transport, base_url="http://test", headers=headers) as b:
            yield a, b

    if _main._run_store is not None:
        await _main._run_store._conn.close()


async def test_listing_shows_only_own_threads(two_clients):
    alice, bob = two_clients
    await alice.post("/chat/stream", json={"message": "alice secret"})
    await bob.post("/chat/stream", json={"message": "bob secret"})

    alice_threads = (await alice.get("/threads")).json()["threads"]
    bob_threads = (await bob.get("/threads")).json()["threads"]

    assert len(alice_threads) == 1
    assert len(bob_threads) == 1
    assert alice_threads[0]["thread_id"] != bob_threads[0]["thread_id"]
    assert "alice secret" in alice_threads[0]["title"]
    assert "bob secret" in bob_threads[0]["title"]


async def test_reading_another_visitors_thread_is_404(two_clients):
    alice, bob = two_clients
    response = await alice.post("/chat/stream", json={"message": "alice secret"})
    alice_thread = thread_id_from(response)

    assert (await alice.get(f"/threads/{alice_thread}")).status_code == 200
    assert (await bob.get(f"/threads/{alice_thread}")).status_code == 404


async def test_cancelling_another_visitors_thread_is_not_found(two_clients):
    alice, bob = two_clients
    response = await alice.post("/chat/stream", json={"message": "alice secret"})
    alice_thread = thread_id_from(response)

    body = (await bob.post(f"/cancel/{alice_thread}")).json()
    assert body["status"] == "not_found"


async def test_thread_read_fails_closed_without_a_run_store(client_with_fake, monkeypatch):
    """No run store means no way to verify ownership, so nothing is served."""
    client, _ = client_with_fake
    response = await client.post("/chat/stream", json={"message": "hello"})
    thread_id = thread_id_from(response)

    async def _no_store():
        return None

    monkeypatch.setattr(_main, "_get_run_store", _no_store)
    assert (await client.get(f"/threads/{thread_id}")).status_code == 404
