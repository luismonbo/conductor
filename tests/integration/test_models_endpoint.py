"""GET /models: fake backend yields an empty list; helper sorts SDK pages."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from harness.api.main import app, _list_model_ids


def test_models_empty_on_fake_backend(monkeypatch):
    # default must be None too, not just models=[] — a non-null default is a
    # profile name the frontend will send back as a live per-call override
    # (ChatRequest.model -> model_override), and direct backends (fake, azure)
    # have no concept of named profiles. HARNESS_DEFAULT_MODEL is set here to
    # prove the backend itself suppresses it, not merely that the var is unset.
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_DEFAULT_MODEL", "local-gemma")
    with TestClient(app) as client:
        resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"models": [], "default": None}


async def test_list_model_ids_sorts():
    class _Models:
        async def list(self):
            return SimpleNamespace(
                data=[SimpleNamespace(id="gpt"), SimpleNamespace(id="claude")]
            )

    stub = SimpleNamespace(models=_Models())
    assert await _list_model_ids(stub) == ["claude", "gpt"]
