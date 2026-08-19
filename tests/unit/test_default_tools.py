"""Unit tests for build_registry()'s tool composition."""
from __future__ import annotations

import pytest

from harness.adapters.embedding.fake import FakeEmbedder
from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.agents.default.tools import build_registry
from harness.core.rag.serve import Retriever


def test_registry_without_retriever_omits_search_documents():
    registry = build_registry()
    assert "search_documents" not in registry.names()


@pytest.mark.asyncio
async def test_registry_with_retriever_includes_search_documents():
    store = InMemoryVectorStore()
    retriever = Retriever(embedder=FakeEmbedder(dimension=4), vector_store=store)

    registry = build_registry(
        retriever=retriever, vector_store=store, default_collection="papers",
    )

    assert "search_documents" in registry.names()


def test_registry_always_includes_existing_tools():
    registry = build_registry()
    assert {"calculator", "recall", "remember"} <= set(registry.names())
