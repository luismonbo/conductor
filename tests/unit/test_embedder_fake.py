from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder


@pytest.mark.asyncio
async def test_same_text_produces_same_vector():
    embedder = FakeEmbedder(dimension=8)
    a = await embedder.embed(["hello world"])
    b = await embedder.embed(["hello world"])
    assert a == b


@pytest.mark.asyncio
async def test_different_text_produces_different_vector():
    embedder = FakeEmbedder(dimension=8)
    [a] = await embedder.embed(["hello"])
    [b] = await embedder.embed(["goodbye"])
    assert a != b


@pytest.mark.asyncio
async def test_batch_embed_preserves_order_and_dimension():
    embedder = FakeEmbedder(dimension=4)
    vectors = await embedder.embed(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)
