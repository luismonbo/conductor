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


async def test_model_field_ignored_for_azure_backend(client_with_fake_azure_backend):
    # A client-supplied model must not reach a direct-credentialed backend:
    # it has one specific, fixed, real deployment configured, and honoring
    # an arbitrary override silently redirects requests to a different
    # deployment name under the same credentials -- unlike openai_compatible
    # (routes per-model through a proxy, by design) or fake (records the
    # value but never calls anything real), azure has no safe destination
    # for a name it wasn't configured with. See docs/devlog/010.
    client, fake = client_with_fake_azure_backend
    resp = await client.post(
        "/chat/stream", json={"message": "hi", "model": "some-other-deployment"}
    )
    assert resp.status_code == 200
    assert fake.requested_models == [None]
