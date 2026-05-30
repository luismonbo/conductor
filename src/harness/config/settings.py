"""Settings + backend selection.

Backend choice (azure | fake | ollama) and memory choice (in_memory | qdrant)
are config, not code. Profiles in config/profiles/ override these for
dev-azure vs edge-pi. This is where the 'swap is a config change, not a
rewrite' promise is actually cashed in.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env")

    # Backend selection
    llm_backend: str = "fake"          # fake | azure | ollama
    memory_backend: str = "in_memory"  # in_memory | qdrant
    tool_parser: str = "native"        # native | prompted

    # Azure OpenAI
    azure_endpoint: str = ""
    azure_deployment: str = "gpt-5.4-mini"
    azure_api_version: str = "2024-10-21"
    azure_api_key: str = ""            # empty -> managed identity

    # Ollama (Gemma-4 path)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Agent
    max_iterations: int = 8
    system_prompt: str = (
        "You are a helpful assistant. Use tools when they help. "
        "When you have the answer, reply directly without calling a tool."
    )


def get_settings() -> Settings:
    return Settings()
