"""Pin the RAG embedding / vector store settings and their defaults."""
from __future__ import annotations

from harness.config.settings import Settings

_RAG_ENV_KEYS = [
    "HARNESS_EMBEDDING_BACKEND",
    "HARNESS_EMBEDDING_BASE_URL",
    "HARNESS_EMBEDDING_MODEL",
    "HARNESS_EMBEDDING_API_KEY",
    "HARNESS_EMBEDDING_DIMENSION",
    "HARNESS_RAG_COLLECTION",
    "HARNESS_RAG_VECTOR_STORE_BACKEND",
    "HARNESS_PGVECTOR_URL",
    "HARNESS_PGVECTOR_TABLE",
    "HARNESS_MILVUS_URI",
    "HARNESS_MILVUS_COLLECTION",
]


def test_rag_settings_have_sensible_defaults(monkeypatch):
    # Isolate from any ambient HARNESS_* env *and* from the developer's local
    # .env, so this asserts the real code defaults. Without _env_file=None this
    # test passes or fails depending on what happens to be in .env.
    for key in _RAG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.embedding_backend == "fake"
    assert settings.embedding_model == "nomic-embed-text-v1.5"
    assert settings.embedding_dimension == 768
    assert settings.pgvector_table == "rag_chunks"
    assert settings.milvus_collection == "rag_chunks"
    assert settings.milvus_uri == "./data/milvus_papers.db"
    assert settings.rag_collection == "papers"
    assert settings.rag_vector_store_backend == "pgvector"


def test_rag_settings_are_overridable_via_env(monkeypatch):
    monkeypatch.setenv("HARNESS_EMBEDDING_BACKEND", "openai_compatible")
    monkeypatch.setenv("HARNESS_EMBEDDING_MODEL", "bge-small-en-v1.5")

    settings = Settings(_env_file=None)

    assert settings.embedding_backend == "openai_compatible"
    assert settings.embedding_model == "bge-small-en-v1.5"
