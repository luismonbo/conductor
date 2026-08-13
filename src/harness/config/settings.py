"""Settings + backend selection.

Backend choice (fake | openai_compatible | azure) and memory choice
(in_memory | pgvector) are config, not code. Profiles in config/profiles/ override these for
dev-azure vs edge-pi. This is where the 'swap is a config change, not a
rewrite' promise is actually cashed in.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env is shared with docker-compose (litellm/langfuse
    # container config) and main.py's direct os.environ reads for Langfuse —
    # those keys aren't HARNESS_-prefixed and must not fail validation here.
    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env", extra="ignore")

    # Backend selection
    llm_backend: str = "fake"          # fake | openai_compatible | azure
    memory_backend: str = "sqlite"      # in_memory | sqlite | pgvector
    tool_parser: str = "native"        # native | prompted

    # OpenAI-compatible local server (llama.cpp llama-server / vLLM / etc.)
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "gemma4:a2b"
    llm_api_key: str = ""              # most local servers ignore this

    # Default model profile sent to the proxy when no pin/override applies.
    # Supersedes llm_model when set; llm_model remains the fallback.
    default_model: str = ""

    # Azure OpenAI
    azure_endpoint: str = ""
    azure_deployment: str = "gpt-5.4-mini"
    azure_api_version: str = "2024-10-21"
    azure_api_key: str = ""            # empty -> managed identity

    # pgvector long-term memory (Postgres DSN; may equal the checkpointer DSN).
    # Empty until the pgvector backend is wired in Phase 5.
    memory_url: str = ""

    # RAG — embedding backend (mirrors llm_backend's swap pattern)
    embedding_backend: str = "fake"        # fake | openai_compatible | azure
    embedding_base_url: str = "http://localhost:8081/v1"  # second llama-server, embedding model
    embedding_model: str = "nomic-embed-text-v1.5"
    embedding_api_key: str = ""
    embedding_dimension: int = 768         # must match the store schema; changing this forces a reindex

    # RAG — retrieval. per_document_k > 0 enables a per-document quota, which
    # stops one large document monopolising top-k on a mixed-corpus query.
    # 0 disables it (plain flat top-k). Default 2 is measured, not guessed: on
    # the papers corpus it lifts recall@5 from 0.769 to 0.808 with no case
    # regressing — the same recall a flat k=10 reaches, at half the prompt cost.
    rag_k: int = 5
    rag_per_document_k: int = 2
    rag_overfetch: int = 5

    # RAG — vector stores (distinct from HARNESS_MEMORY_BACKEND / HARNESS_MEMORY_URL)
    rag_collection: str = "papers"
    pgvector_url: str = ""                 # Postgres DSN; may equal checkpointer/memory DSN
    pgvector_table: str = "rag_chunks"
    milvus_uri: str = "./data/milvus_papers.db"
    milvus_collection: str = "rag_chunks"

    # Checkpointer
    checkpointer: str = "sqlite"       # memory | sqlite | postgres
    checkpointer_url: str = "./harness.sqlite"

    # Agent
    agent: str = "default"
    max_iterations: int = 8
    system_prompt: str = (
        "You are a helpful assistant. "
        "Use the calculator tool for any arithmetic operation or numeric "
        "computation — even when the answer seems obvious. "
        "For questions about the user's personal information (name, location, "
        "job title, preferences, ongoing projects), always call the recall "
        "tool; never guess. "
        "For world knowledge questions (facts, history, science), answer directly."
    )


def get_settings() -> Settings:
    return Settings()
