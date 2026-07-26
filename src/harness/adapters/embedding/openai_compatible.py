"""Embeddings via any OpenAI-compatible /v1/embeddings endpoint — locally a
second llama-server instance serving an embedding model (recommend
nomic-embed-text-v1.5, 768-dim). Same base_url-keyed swap pattern as
adapters/llm/openai_compatible.py."""
from __future__ import annotations

from openai import AsyncOpenAI


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, model: str, api_key: str = "") -> None:
        self._model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")

    @property
    def model_id(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
