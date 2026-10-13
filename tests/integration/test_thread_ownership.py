"""A visitor sees only their own threads and cannot read anyone else's."""
import asyncio

import aiosqlite
import httpx2
import pytest

import harness.api.main as _main
from harness.adapters.llm.fake import FakeLLMClient
from harness.core.types import LLMResponse, ToolCall
from harness.observability.run_store import RunStore
from tests.integration.conftest import _TEST_API_KEY, sse_frames, thread_id_from


@pytest.fixture(params=["sqlite", "memory"])
async def two_clients(request, monkeypatch, tmp_path):
    """Two independent cookie jars against one app, once per checkpointer mode.

    Ownership must hold identically in every mode — memory mode has its own
    in-process run store — so every test using this runs against both.
    """
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_CHECKPOINTER", request.param)
    monkeypatch.setenv("HARNESS_CHECKPOINTER_URL", str(tmp_path / "h.sqlite"))

    import harness.orchestration.build as build_module

    fake = FakeLLMClient([LLMResponse(text="hi")] * 10)
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)

    transport = httpx2.ASGITransport(app=_main.app)
    headers = {"Authorization": f"Bearer {_TEST_API_KEY}"}
    async with httpx2.AsyncClient(transport=transport, base_url="https://test", headers=headers) as a:
        async with httpx2.AsyncClient(transport=transport, base_url="https://test", headers=headers) as b:
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


async def test_resuming_another_visitors_thread_is_404(two_clients, monkeypatch):
    """Bob cannot answer Alice's pending approval; she still can, afterwards.

    The run pauses on a real interrupt (a proposed `remember`; conftest enables
    the memory tools), so Bob is refused while there is something to answer,
    and Alice answering after him proves his attempt left it pending. Both
    decisions deny, so no tool runs even if the guard regresses.
    """
    alice, bob = two_clients

    import harness.orchestration.build as build_module

    proposal = LLMResponse(
        text="",
        tool_calls=(ToolCall(id="c1", name="remember", arguments={"text": "alice secret"}),),
    )
    fake = FakeLLMClient([proposal, LLMResponse(text="Not saved.")])
    monkeypatch.setattr(build_module, "build_llm", lambda settings, parser: fake)

    response = await alice.post("/chat/stream", json={"message": "remember my secret"})
    assert "interrupt" in [f.get("type") for f in sse_frames(response)]
    alice_thread = thread_id_from(response)

    deny = {"decision": {"action": "deny"}}
    assert (await bob.post(f"/resume/{alice_thread}", json=deny)).status_code == 404

    own = await alice.post(f"/resume/{alice_thread}", json=deny)
    assert own.status_code == 200
    assert sse_frames(own)[-1]["type"] == "final"


async def test_a_failed_run_store_setup_is_retried_not_cached(monkeypatch):
    """A store cached before its schema is in place would 404 every continued
    thread until restart, so a failed setup is discarded (and closed)."""
    real_create_table = RunStore.create_table
    attempts: list[RunStore] = []

    async def _fails_once(self):
        attempts.append(self)
        if len(attempts) == 1:
            raise aiosqlite.OperationalError("database is locked")
        await real_create_table(self)

    monkeypatch.setattr(RunStore, "create_table", _fails_once)

    with pytest.raises(aiosqlite.OperationalError):
        await _main._get_run_store()
    assert _main._run_store is None
    with pytest.raises(ValueError):
        await attempts[0]._conn.execute("SELECT 1")  # its connection was closed

    store = await _main._get_run_store()
    await store.start_run("r1", "t1", "default", "fake", "sid-a")
    assert await store.owns_thread("sid-a", "t1") is True
