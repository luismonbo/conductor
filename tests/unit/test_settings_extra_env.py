"""Settings must tolerate unrelated env vars living alongside HARNESS_* ones.

.env is shared with docker-compose (litellm/langfuse container config) and
with main.py's direct os.environ reads for Langfuse — none of those keys are
HARNESS_-prefixed, and Settings must not choke on their presence.
"""
from harness.config.settings import Settings


def test_settings_ignores_non_harness_env_vars(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-litellm-master")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-example")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("NEXTAUTH_SECRET", "deadbeef")

    settings = Settings()

    assert settings.llm_backend == "fake"
