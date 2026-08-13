"""GET /models: fake backend yields an empty list; helper sorts SDK pages."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from harness.api.main import app, _list_model_ids


def test_models_empty_on_fake_backend(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_BACKEND", "fake")
    monkeypatch.setenv("HARNESS_DEFAULT_MODEL", "local-gemma")
    with TestClient(app) as client:
        resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"models": [], "default": "local-gemma"}


async def test_list_model_ids_sorts():
    class _Models:
        async def list(self):
            return SimpleNamespace(
                data=[SimpleNamespace(id="gpt"), SimpleNamespace(id="claude")]
            )

    stub = SimpleNamespace(models=_Models())
    assert await _list_model_ids(stub) == ["claude", "gpt"]
