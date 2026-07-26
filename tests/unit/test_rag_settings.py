from __future__ import annotations

from harness.config.settings import Settings


def test_rag_settings_have_sensible_defaults():
    settings = Settings()
    assert settings.embedding_backend == "fake"
    assert settings.embedding_model == "nomic-embed-text-v1.5"
    assert settings.embedding_dimension == 768
    assert settings.pgvector_table == "rag_chunks"
    assert settings.milvus_collection == "rag_chunks"
    assert settings.milvus_uri == "./data/milvus_papers.db"
    assert settings.rag_collection == "papers"


def test_rag_settings_are_overridable_via_env(monkeypatch):
    monkeypatch.setenv("HARNESS_EMBEDDING_BACKEND", "openai_compatible")
    monkeypatch.setenv("HARNESS_EMBEDDING_MODEL", "bge-small-en-v1.5")
    settings = Settings()
    assert settings.embedding_backend == "openai_compatible"
    assert settings.embedding_model == "bge-small-en-v1.5"
