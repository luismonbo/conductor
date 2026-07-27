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
        # One sha256 digest is 32 bytes, so slicing it can never fill the
        # default 768-dim vector -- it would quietly emit 32 floats and every
        # real store would reject the write for mismatching its schema.
        # Hash with a counter until the requested width is covered.
        values: list[float] = []
        block = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(f"{block}:{text}".encode()).digest()
            values.extend(b / 255.0 for b in digest)
            block += 1
        return values[: self._dimension]
