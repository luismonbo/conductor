"""POST /chat/stream {model} lands in the LLM call as the request override."""
from tests.integration.conftest import sse_frames


async def test_model_field_reaches_llm(client_with_fake):
    client, fake = client_with_fake
    resp = await client.post(
        "/chat/stream", json={"message": "hi", "model": "claude"}
    )
    assert resp.status_code == 200
    assert any(f.get("type") == "final" for f in sse_frames(resp))
    assert fake.requested_models == ["claude"]


async def test_missing_model_field_passes_none(client_with_fake):
    client, fake = client_with_fake
    resp = await client.post("/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert fake.requested_models == [None]
