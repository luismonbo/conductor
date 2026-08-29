"""Unit tests for RAG-related factory functions in orchestration/build.py."""
from __future__ import annotations

from pathlib import Path

import yaml

from harness.adapters.vectorstore.in_memory import InMemoryVectorStore
from harness.config.settings import Settings
from harness.core.rag.serve import DiversifiedRetriever, Retriever
from harness.orchestration.build import build_retriever, list_collections


def test_list_collections_empty_directory_returns_empty_list(tmp_path: Path):
    assert list_collections(tmp_path) == []


def test_list_collections_reads_collection_field_from_each_manifest(tmp_path: Path):
    (tmp_path / "papers.yaml").write_text(yaml.safe_dump({"collection": "papers"}))
    (tmp_path / "manuals.yaml").write_text(yaml.safe_dump({"collection": "manuals"}))

    assert list_collections(tmp_path) == ["manuals", "papers"]


def test_list_collections_missing_directory_returns_empty_list(tmp_path: Path):
    assert list_collections(tmp_path / "does_not_exist") == []


def test_build_retriever_returns_plain_retriever_when_quota_disabled():
    settings = Settings(
        _env_file=None,
        embedding_backend="fake",
        embedding_dimension=768,
        rag_per_document_k=0,
        api_key="test-key",
    )
    store = InMemoryVectorStore()

    retriever = build_retriever(settings, store)

    assert type(retriever) is Retriever


def test_build_retriever_wraps_in_diversified_retriever_when_quota_enabled():
    settings = Settings(
        _env_file=None,
        embedding_backend="fake",
        embedding_dimension=768,
        rag_per_document_k=2,
        rag_overfetch=5,
        api_key="test-key",
    )
    store = InMemoryVectorStore()

    retriever = build_retriever(settings, store)

    assert isinstance(retriever, DiversifiedRetriever)
