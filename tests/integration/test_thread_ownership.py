"""A visitor sees only their own threads and cannot read anyone else's."""
import asyncio

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


async def test_chat_stream_rejects_another_visitors_thread_id(two_clients):
    """A caller may continue its own thread but not name someone else's.

    Without the check, Bob's post would load Alice's history into his
    response AND record his session as an owner of her thread.
    """
    alice, bob = two_clients
    response = await alice.post("/chat/stream", json={"message": "alice secret"})
    alice_thread = thread_id_from(response)

    hijack = await bob.post("/chat/stream", json={"message": "recap", "thread_id": alice_thread})
    assert hijack.status_code == 404
    # No shared ownership was recorded as a side effect of the attempt.
    assert (await bob.get(f"/threads/{alice_thread}")).status_code == 404

    # The owner can still continue her own thread through the same check.
    own = await alice.post("/chat/stream", json={"message": "and again", "thread_id": alice_thread})
    assert own.status_code == 200


async def test_cancelling_another_visitors_thread_is_not_found(two_clients, monkeypatch):
    """Bob is denied; Alice, cancelling the same in-flight run, succeeds.

    Both directions matter: a single not_found assertion would also pass
    against an unguarded cancel_run if the run had already finished by the
    time we checked — proving nothing about ownership. The second assertion
    (Alice succeeds) is what proves the guard discriminates by owner rather
    than blanket-denying.

    httpx2's ASGITransport fully drains the ASGI app (runs chat_stream's
    background task to completion) before returning anything at all to the
    caller — client.stream() included, since ASGIResponseStream just replays
    an already-fully-collected buffer. So there is no way to observe an
    in-flight run from a client response. Instead: run the POST as a
    concurrent task, and use a real asyncio.Event to pause the fake LLM call
    mid-run, so the task is genuinely still registered in `_running` — found
    by reading it directly, the same way cancel_run itself would look it up
    — while we issue the two cancel requests concurrently on the same loop.

    For the same reason, each client's sid cookie must be primed with a plain
    request before the paused POST: the paused request's own Set-Cookie
    header cannot reach the client's jar until its response returns, which
    ASGITransport withholds until the run finishes — long after the cancel
    requests need it.
    """
    alice, bob = two_clients

    # Each client needs its sid cookie BEFORE the paused chat request: that
    # request's Set-Cookie cannot reach the client's jar until its response
    # returns, which httpx2's ASGITransport withholds until the run finishes.
    await alice.get("/health")
    await bob.get("/health")

    started = asyncio.Event()
    release = asyncio.Event()

    class _PausableFakeLLMClient(FakeLLMClient):
        async def stream(self, messages, tools=None, model=None):
            started.set()
            await release.wait()
            async for chunk in super().stream(messages, tools=tools, model=model):
                yield chunk

    import harness.orchestration.build as build_module

    monkeypatch.setattr(
        build_module,
        "build_llm",
        lambda settings, parser: _PausableFakeLLMClient([LLMResponse(text="hi")]),
    )

    task = asyncio.create_task(alice.post("/chat/stream", json={"message": "alice secret"}))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert list(_main._running.keys()), "run never registered in _running"
        alice_thread = next(iter(_main._running))

        assert (await bob.post(f"/cancel/{alice_thread}")).json()["status"] == "not_found"
        assert (await alice.post(f"/cancel/{alice_thread}")).json()["status"] == "cancelled"
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)


async def test_resuming_another_visitors_thread_is_404(two_clients):
    alice, bob = two_clients
    response = await alice.post("/chat/stream", json={"message": "alice secret"})
    alice_thread = thread_id_from(response)

    resp = await bob.post(f"/resume/{alice_thread}", json={"decision": {"approved": True}})
    assert resp.status_code == 404


async def test_thread_read_fails_closed_without_a_run_store(client_with_fake, monkeypatch):
    """No run store means no way to verify ownership, so nothing is served."""
    client, _ = client_with_fake
    response = await client.post("/chat/stream", json={"message": "hello"})
    thread_id = thread_id_from(response)

    async def _no_store():
        return None

    monkeypatch.setattr(_main, "_get_run_store", _no_store)
    assert (await client.get(f"/threads/{thread_id}")).status_code == 404


async def test_resume_fails_closed_without_a_run_store(client_with_fake, monkeypatch):
    """No run store means no way to verify ownership, so resume is refused too."""
    client, _ = client_with_fake
    response = await client.post("/chat/stream", json={"message": "hello"})
    thread_id = thread_id_from(response)

    async def _no_store():
        return None

    monkeypatch.setattr(_main, "_get_run_store", _no_store)
    resp = await client.post(f"/resume/{thread_id}", json={"decision": {"approved": True}})
    assert resp.status_code == 404
