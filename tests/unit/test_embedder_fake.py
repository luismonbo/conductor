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


@pytest.mark.asyncio
async def test_dimension_larger_than_one_hash_digest_is_honoured():
    """A sha256 digest is 32 bytes. The default embedding_dimension is 768, so
    a fake that slices a single digest silently emits 32-dim vectors and every
    real store rejects them for mismatching its schema."""
    embedder = FakeEmbedder(dimension=768)
    [vector] = await embedder.embed(["hello"])
    assert len(vector) == 768


@pytest.mark.asyncio
async def test_large_vectors_are_still_deterministic_and_text_dependent():
    embedder = FakeEmbedder(dimension=768)
    [a] = await embedder.embed(["hello"])
    [b] = await embedder.embed(["hello"])
    [c] = await embedder.embed(["goodbye"])
    assert a == b
    assert a != c
