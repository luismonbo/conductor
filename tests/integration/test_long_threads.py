"""A long thread is answered: no message count ends a conversation."""
from __future__ import annotations

from harness.api.rate_limit import limiter
from tests.integration.conftest import sse_frames, thread_id_from


async def test_a_21st_message_in_a_thread_is_answered(client_with_fake):
    client, _ = client_with_fake
    first = await client.post("/chat/stream", json={"message": "m1"})
    thread_id = thread_id_from(first)
    for i in range(2, 22):
        limiter.reset()  # 21 messages in a row trip the strict per-minute limit otherwise
        reply = await client.post(
            "/chat/stream", json={"message": f"m{i}", "thread_id": thread_id}
        )
        assert sse_frames(reply)[-1]["stopped_reason"] == "final_answer", f"message {i}"
