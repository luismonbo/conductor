"""Settings + backend selection.

Backend choice (fake | openai_compatible | azure) and memory choice
(in_memory | pgvector) are config, not code. Profiles in config/profiles/ override these for
dev-azure vs edge-pi. This is where the 'swap is a config change, not a
rewrite' promise is actually cashed in.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env")

    # Backend selection
    llm_backend: str = "fake"          # fake | openai_compatible | azure
    memory_backend: str = "in_memory"  # in_memory | pgvector
    tool_parser: str = "native"        # native | prompted

    # OpenAI-compatible local server (llama.cpp llama-server / vLLM / etc.)
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "gemma4:a2b"
    llm_api_key: str = ""              # most local servers ignore this

    # Azure OpenAI
    azure_endpoint: str = ""
    azure_deployment: str = "gpt-5.4-mini"
    azure_api_version: str = "2024-10-21"
    azure_api_key: str = ""            # empty -> managed identity

    # pgvector long-term memory (Postgres DSN; may equal the checkpointer DSN).
    # Empty until the pgvector backend is wired in Phase 5.
    memory_url: str = ""

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
