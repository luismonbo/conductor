"""Constructing many MilvusStores over one database file, in one process.

RagRunner builds a fresh RagPipeline per eval case, so a 15-case run
constructs 15 MilvusStores over the same Milvus Lite file. This pins that
access pattern as safe.

The constraint that *is* real is concurrency across processes: Milvus Lite is
an embedded single-process database, and two processes opening the same file
fail with `ConnectionConfigException: Open local milvus failed`. Running two
eval processes against the same `--vector-store milvus` concurrently will do
exactly that. Sequential cross-process access is covered by
test_milvus_reopen.py.
"""
from __future__ import annotations

import pytest

from harness.core.rag.document import Chunk

pytest.importorskip("milvus_lite", reason="milvus_lite extra not installed")

from harness.adapters.vectorstore.milvus_store import MilvusStore  # noqa: E402


@pytest.mark.asyncio
async def test_many_stores_over_one_file_keep_working(tmp_path):
    uri = str(tmp_path / "repeated.db")

    seed = MilvusStore(uri=uri, collection="rag_chunks", vector_size=3)
    await seed.upsert(
        [Chunk(chunk_id="d1:0", document_id="d1", collection="papers",
               text="self-attention", section_path=("Method",))],
        [[1.0, 0.0, 0.0]],
    )

    # One fresh store per "eval case", well past where the real run broke.
    for i in range(20):
        store = MilvusStore(uri=uri, collection="rag_chunks", vector_size=3)
        results = await store.search([1.0, 0.0, 0.0], k=3)
        assert results, f"search returned nothing on store #{i}"
        assert results[0].chunk.chunk_id == "d1:0"
