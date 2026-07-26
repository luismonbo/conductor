from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness.adapters.embedding.openai_compatible import OpenAICompatibleEmbedder


@pytest.mark.asyncio
async def test_embed_calls_embeddings_endpoint_and_returns_vectors_in_order():
    embedder = OpenAICompatibleEmbedder(
        base_url="http://localhost:8081/v1", model="nomic-embed-text-v1.5"
    )

    fake_response = type(
        "Resp", (),
        {"data": [type("Item", (), {"embedding": [0.1, 0.2]})(), type("Item", (), {"embedding": [0.3, 0.4]})()]},
    )()

    with patch.object(
        embedder._client.embeddings, "create", new=AsyncMock(return_value=fake_response)
    ) as mock_create:
        vectors = await embedder.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_create.assert_awaited_once_with(model="nomic-embed-text-v1.5", input=["hello", "world"])
