"""Threads list and transcript reload from checkpointer + run store."""
from tests.integration.conftest import thread_id_from


async def test_threads_lists_conversation_with_title(client_with_fake):
    client, _ = client_with_fake
    resp = await client.post("/chat/stream", json={"message": "remember the milk"})
    assert resp.status_code == 200

    listing = (await client.get("/threads")).json()
    assert len(listing["threads"]) == 1
    thread = listing["threads"][0]
    assert thread["title"].startswith("remember the milk")
    assert thread["runs"] == 1


async def test_thread_messages_roundtrip(client_with_fake):
    client, _ = client_with_fake
    resp = await client.post("/chat/stream", json={"message": "hello there"})
    thread_id = thread_id_from(resp)  # first SSE frame carries it

    body = (await client.get(f"/threads/{thread_id}")).json()
    roles = [m["role"] for m in body["messages"]]
    assert roles[0] == "user"
    assert "assistant" in roles
    assert all(m["role"] != "system" for m in body["messages"])


async def test_thread_messages_404_on_unknown(client_with_fake):
    client, _ = client_with_fake
    resp = await client.get("/threads/does-not-exist")
    assert resp.status_code == 404
