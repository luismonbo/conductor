"""Langfuse callbacks: off without keys, on with keys, never fatal."""
import harness.api.main as main


def test_no_keys_no_callbacks(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    callbacks, metadata = main._build_callbacks("t1", "default")
    assert callbacks == []
    assert metadata == {}


def test_keys_present_builds_handler(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:59999")  # nothing listens
    callbacks, metadata = main._build_callbacks("t1", "default")
    # Handler construction is offline; network only happens on flush (batched).
    assert len(callbacks) == 1
    assert metadata == {
        "langfuse_session_id": "t1",
        "langfuse_tags": ["default"],
    }


def test_import_failure_is_swallowed(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    def _boom(*a, **k):
        raise RuntimeError("no langfuse for you")

    monkeypatch.setattr(main, "_make_langfuse_handler", _boom)
    callbacks, metadata = main._build_callbacks("t1", "default")
    assert callbacks == [] and metadata == {}
