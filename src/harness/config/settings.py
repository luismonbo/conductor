"""Settings + backend selection.

Backend choice (fake | openai_compatible | azure) and memory choice
(in_memory | pgvector) are config, not code. Deployment targets in
config/targets/ (selected via HARNESS_TARGET) override these per-deployment
— e.g. the daily driver vs. a production site. This is where the 'swap is a
config change, not a rewrite' promise is actually cashed in.
"""
from __future__ import annotations

import os

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    SettingsConfigDict,
)

from harness.config.targets import resolve_target_overrides


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

    # Azure embeddings deployment name (azure_endpoint/azure_api_version/azure_api_key
    # above are shared with the chat backend — same Azure resource, different deployment).
    azure_embedding_deployment: str = ""

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
    rag_vector_store_backend: str = "pgvector"  # pgvector | milvus | in_memory
    pgvector_url: str = ""                 # Postgres DSN; may equal checkpointer/memory DSN
    pgvector_table: str = "rag_chunks"
    milvus_uri: str = "./data/milvus_papers.db"
    milvus_collection: str = "rag_chunks"

    # Checkpointer
    checkpointer: str = "sqlite"       # memory | sqlite | postgres
    checkpointer_url: str = "./harness.sqlite"

    # The entry agent: a folder under src/harness/agents/. Its prompt.py holds
    # the prompt; prompts never come from config.
    agent: str = "default"
    max_iterations: int = 8

    # Long-term memory is a single shared store with no per-visitor partition,
    # so remember/recall must stay off on any public deployment. Default-safe;
    # a single-operator deployment opts in via its target file or
    # HARNESS_MEMORY_TOOLS_ENABLED=true. See
    # docs/superpowers/specs/2026-09-19-visitor-scoped-threads-design.md.
    memory_tools_enabled: bool = False

    # Rate limiting — see docs/superpowers/specs/2026-08-27-rate-limiting-design.md.
    # Strict tier covers the LLM-invoking endpoints (the cost driver if abused);
    # default tier covers everything else except /health, which is never limited.
    rate_limit_enabled: bool = True
    rate_limit_strict: str = "15/minute"
    rate_limit_default: str = "60/minute"

    # Authentication — see docs/superpowers/specs/2026-08-28-authentication-design.md.
    # A single shared key gates every route except /health (and FastAPI's own
    # /docs, /openapi.json, /redoc).
    auth_enabled: bool = True
    api_key: str = ""

    # Visitor session cookie (sid): Secure by default. Set false only for local
    # development on a browser that does not treat http://localhost as a
    # secure context.
    session_cookie_secure: bool = True

    # Input guard (SL-2) — see docs/superpowers/specs/2026-09-27-input-guard-design.md.
    # input_guard_enabled gates only the TypeSafe (Jev) checks and fails closed
    # without a key (validator below). The four limits are independent of it
    # and each is off at 0. Defaults are the public website's values; a
    # single-operator deployment turns all of it off in its target file.
    input_guard_enabled: bool = True
    typesafe_api_key: str = ""
    guard_model: str = "jev-1.13.0"  # pinned: thresholds are calibrated against it
    guard_timeout_seconds: float = Field(3.0, gt=0)
    max_message_chars: int = Field(2000, ge=0)
    max_output_tokens: int = Field(1500, ge=0)  # per LLM call
    max_thread_turns: int = Field(20, ge=0)  # runs per thread
    daily_token_budget: int = Field(33000, ge=0)  # all visitors per UTC day: monthly quota / 30

    @model_validator(mode="after")
    def _require_api_key_when_auth_enabled(self) -> "Settings":
        if self.auth_enabled and not self.api_key:
            raise ValueError(
                "HARNESS_AUTH_ENABLED is true but HARNESS_API_KEY is empty. "
                "Set HARNESS_API_KEY, or set HARNESS_AUTH_ENABLED=false."
            )
        return self

    @model_validator(mode="after")
    def _require_typesafe_key_when_guard_enabled(self) -> "Settings":
        if self.input_guard_enabled and not self.typesafe_api_key:
            raise ValueError(
                "HARNESS_INPUT_GUARD_ENABLED is true but HARNESS_TYPESAFE_API_KEY is "
                "empty. Set HARNESS_TYPESAFE_API_KEY, or set "
                "HARNESS_INPUT_GUARD_ENABLED=false."
            )
        return self


# Removed settings and where their job went. Settings ignores unknown
# variables (extra="ignore"), so a leftover one would silently do nothing.
_REMOVED_ENV_VARS = {
    "HARNESS_SYSTEM_PROMPT": (
        "HARNESS_SYSTEM_PROMPT was removed: the prompt lives in "
        "src/harness/agents/<name>/prompt.py and HARNESS_AGENT picks the agent."
    ),
}


def _reject_removed_env_vars(env_file: str = ".env") -> None:
    dotenv = dotenv_values(env_file) if os.path.isfile(env_file) else {}
    for name, message in _REMOVED_ENV_VARS.items():
        if name in os.environ or name in dotenv:
            raise ValueError(message)


def get_settings() -> Settings:
    _reject_removed_env_vars()
    target_name = os.environ.get("HARNESS_TARGET", "")
    if not target_name:
        return Settings()

    overrides = resolve_target_overrides(target_name)
    unknown_fields = overrides.keys() - Settings.model_fields.keys()
    if unknown_fields:
        raise ValueError(
            f"Target {target_name!r} sets unknown field(s): {sorted(unknown_fields)}"
        )

    # A field the environment sets (a real env var or .env, even to the field's
    # own default) keeps that value; the target file fills in the rest.
    # Settings is built once, from the merged values, so the fail-closed checks
    # (auth, input guard) see a target file's `false` instead of rejecting the
    # environment-only defaults first.
    explicit = EnvSettingsSource(Settings)().keys() | DotEnvSettingsSource(Settings)().keys()
    return Settings(**{key: value for key, value in overrides.items() if key not in explicit})
