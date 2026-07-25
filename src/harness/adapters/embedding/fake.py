"""Deterministic embedder for tests and local boot without a real model —
same role as adapters/llm/fake.py."""
from __future__ import annotations

import hashlib


class FakeEmbedder:
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "fake-embedder"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self._dimension]]
