"""AzureOpenAIEmbedder must call the deployment's embeddings endpoint and
preserve vector order. Uses the client injection seam (same seam as
AzureOpenAIClient) so no real Azure SDK / credentials are touched."""
from __future__ import annotations

from types import SimpleNamespace

from harness.adapters.embedding.azure_openai import AzureOpenAIEmbedder


class _RecordingEmbeddings:
    def __init__(self, response) -> None:
        self._response = response
        self.kwargs: list[dict] = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return self._response


def _embedder_with_recorder():
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
    )
    embeddings = _RecordingEmbeddings(response)
    embedder = AzureOpenAIEmbedder(
        deployment="text-embedding-3-small",
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-21",
        client=SimpleNamespace(embeddings=embeddings),
    )
    return embedder, embeddings


async def test_embed_calls_embeddings_endpoint_and_returns_vectors_in_order():
    embedder, embeddings = _embedder_with_recorder()

    vectors = await embedder.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert embeddings.kwargs[0] == {"model": "text-embedding-3-small", "input": ["hello", "world"]}


def test_model_id_returns_deployment_name():
    embedder, _ = _embedder_with_recorder()
    assert embedder.model_id == "text-embedding-3-small"
