"""Settings must tolerate unrelated env vars living alongside HARNESS_* ones.

.env is shared with docker-compose (the LiteLLM proxy's master and provider
keys) — none of those keys are HARNESS_-prefixed, and Settings must not choke
on their presence.
"""
from harness.config.settings import Settings


def test_settings_ignores_non_harness_env_vars(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-litellm-master")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-example")

    settings = Settings()

    assert settings.llm_backend == "fake"
