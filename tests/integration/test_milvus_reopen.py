"""Reopening an existing Milvus collection from a *separate process*.

The contract suite always creates a fresh collection in a fresh tmp file, so
the create path (which auto-loads the collection) always runs. A reader that
opens an already-populated store creates nothing — and Milvus refuses to
search a collection that has not been loaded:

    Collection 'rag_chunks' is in state 'released'; call load() before search

Both halves run as subprocesses. Two MilvusStore instances in one process
share the same milvus-lite engine, which keeps the collection loaded and
hides the bug; and milvus-lite holds a file lock, so the parent must not open
the database at all. This is exactly the real CLI shape — `ingest` writes in
one process, `rag_query` reads in another.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("milvus_lite", reason="milvus_lite extra not installed")


_WRITER = textwrap.dedent(
    """
    import asyncio, sys
    from harness.adapters.vectorstore.milvus_store import MilvusStore
    from harness.core.rag.document import Chunk

    store = MilvusStore(uri=sys.argv[1], collection="rag_chunks", vector_size=3)
    chunk = Chunk(
        chunk_id="d1:0", document_id="d1", collection="papers",
        text="the model uses self-attention", section_path=("Method",),
        source_path="papers/attn.pdf",
    )
    asyncio.run(store.upsert([chunk], [[1.0, 0.0, 0.0]]))
    print("WROTE")
    """
)

_READER = textwrap.dedent(
    """
    import asyncio, sys
    from harness.adapters.vectorstore.milvus_store import MilvusStore

    store = MilvusStore(uri=sys.argv[1], collection="rag_chunks", vector_size=3)
    results = asyncio.run(store.search([1.0, 0.0, 0.0], k=3))
    assert results, "no results returned"
    print(f"OK {results[0].chunk.chunk_id} {results[0].chunk.source_path}")
    """
)


def _run(script: str, uri: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script, uri], capture_output=True, text=True, timeout=180
    )


def test_a_fresh_process_can_search_an_existing_collection(tmp_path):
    uri = str(tmp_path / "reopen.db")

    write = _run(_WRITER, uri)
    assert write.returncode == 0, f"writer failed:\n{write.stderr[-2000:]}"

    read = _run(_READER, uri)

    assert read.returncode == 0, f"reader failed:\n{read.stderr[-2000:]}"
    assert "OK d1:0 papers/attn.pdf" in read.stdout
