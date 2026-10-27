"""Settings + backend selection.

Backend choice (fake | openai_compatible | azure) and memory choice
(in_memory | pgvector) are config, not code. Deployment targets in
config/targets/ (selected via HARNESS_TARGET) override these per-deployment
— e.g. the daily driver vs. a production site. This is where the 'swap is a
config change, not a rewrite' promise is actually cashed in.
"""
from __future__ import annotations

import os
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field, field_validator, model_validator
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
    # hide_input_in_errors: a validation error otherwise ends with input_value=, the
    # first and last characters of every setting, and the API keys are among them.
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

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

    # Wiki: the website agent's knowledge, loaded into memory at startup. One
    # source: a local folder, or a private Azure Blob container read with
    # DefaultAzureCredential (the Container App's managed identity). See
    # docs/superpowers/specs/2026-10-13-wiki-search-design.md.
    wiki_dir: str = ""
    wiki_blob_account_url: str = ""  # e.g. https://<account>.blob.core.windows.net
    wiki_blob_container: str = ""
    wiki_blob_prefix: str = ""  # optional folder inside the container, e.g. wiki/

    # Checkpointer and run store (orchestration/storage.py): memory | sqlite | postgres.
    # The URL is the SQLite file, or for postgres a postgresql:// connection string:
    # a secret, which the production target gets from the Container App's environment.
    checkpointer: str = "sqlite"
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

    # Tool arguments and results reach a browser (the SSE stream and
    # GET /threads/{id}) only when true; off, a browser gets each tool's name
    # only. Default-safe for a public site; the owner's own UI turns it on. See
    # security/egress.py and docs/superpowers/specs/2026-10-13-egress-hardening-design.md.
    show_tool_details: bool = False

    # Level of the harness's own log lines (harness.*); every other library
    # stays at WARNING. See observability/logging_setup.py.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

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
    # How long sid lasts. True keeps it for a year, so a personal deployment
    # remembers its visitor. False sends it without a Max-Age, so the browser
    # drops it when it closes: the public site's target, whose privacy notice
    # says so.
    session_cookie_persistent: bool = True

    # Input guard (SL-2) — see docs/superpowers/specs/2026-09-27-input-guard-design.md.
    # input_guard_enabled gates only the Microsoft-Decision-1 checks and fails
    # closed without an Azure endpoint (validator below). The checks run on the
    # Foundry resource azure_endpoint names, with the LLM's credentials
    # (azure_api_key, else the managed identity). The four limits are independent
    # of it and each is off at 0. Defaults are the public website's values; a
    # single-operator deployment turns all of it off in its target file.
    input_guard_enabled: bool = True
    # The deployment's name, not the model's. Thresholds are calibrated against
    # one model version, so pin the version on the deployment (no auto-upgrade).
    guard_deployment: str = "Decision-1"
    guard_timeout_seconds: float = Field(3.0, gt=0)
    max_message_chars: int = Field(2000, ge=0)
    max_output_tokens: int = Field(1500, ge=0)  # per LLM call
    max_thread_turns: int = Field(20, ge=0)  # runs per thread
    daily_token_budget: int = Field(33000, ge=0)  # all visitors per UTC day: monthly quota / 30

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _require_api_key_when_auth_enabled(self) -> "Settings":
        if self.auth_enabled and not self.api_key:
            raise ValueError(
                "HARNESS_AUTH_ENABLED is true but HARNESS_API_KEY is empty. "
                "Set HARNESS_API_KEY, or set HARNESS_AUTH_ENABLED=false."
            )
        return self

    @model_validator(mode="after")
    def _require_a_guard_target_when_guard_enabled(self) -> "Settings":
        # The guard fails open, so a typo here would switch it off silently.
        if not self.input_guard_enabled:
            return self
        if not self.azure_endpoint.startswith(("https://", "http://")):
            raise ValueError(
                "HARNESS_INPUT_GUARD_ENABLED is true but HARNESS_AZURE_ENDPOINT is "
                "empty or not an http(s) URL. Set HARNESS_AZURE_ENDPOINT to the "
                "Foundry resource that has the Decision-1 deployment, or set "
                "HARNESS_INPUT_GUARD_ENABLED=false."
            )
        if not self.guard_deployment.strip():
            raise ValueError(
                "HARNESS_INPUT_GUARD_ENABLED is true but HARNESS_GUARD_DEPLOYMENT is "
                "empty. Set it to the Decision-1 deployment's name, or set "
                "HARNESS_INPUT_GUARD_ENABLED=false."
            )
        return self

    @model_validator(mode="after")
    def _postgres_needs_a_postgresql_url(self) -> "Settings":
        # Without this, a missing secret would leave the SQLite default
        # ./harness.sqlite, which psycopg rejects with an unhelpful parse error.
        if self.checkpointer == "postgres" and not self.checkpointer_url.startswith(
            ("postgresql://", "postgres://")
        ):
            raise ValueError(
                "HARNESS_CHECKPOINTER=postgres needs HARNESS_CHECKPOINTER_URL set to "
                "a postgresql:// connection string"
            )
        return self

    @model_validator(mode="after")
    def _one_wiki_source(self) -> "Settings":
        blob = (self.wiki_blob_account_url, self.wiki_blob_container)
        if any(blob) and not all(blob):
            raise ValueError(
                "Set both HARNESS_WIKI_BLOB_ACCOUNT_URL and HARNESS_WIKI_BLOB_CONTAINER, "
                "or neither."
            )
        if self.wiki_blob_prefix and not all(blob):
            raise ValueError(
                "HARNESS_WIKI_BLOB_PREFIX needs HARNESS_WIKI_BLOB_ACCOUNT_URL and "
                "HARNESS_WIKI_BLOB_CONTAINER."
            )
        if self.wiki_dir and all(blob):
            raise ValueError(
                "Set one wiki source: HARNESS_WIKI_DIR, or the HARNESS_WIKI_BLOB_* "
                "settings, not both."
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
