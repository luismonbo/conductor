"""build_input_guard: limits always; the Decision-1 classifier only when enabled, wired to the Foundry endpoint."""
from __future__ import annotations

import json
import time

import httpx
import pytest
from azure.core.credentials import AccessToken

from harness.config.settings import Settings
from harness.orchestration.build import build_input_guard
from harness.security.input_guard import GuardContext, GuardLimits

ENDPOINT = "https://foundry.example"
CHECK_IDS = ("extraction", "private_info", "off_topic")


@pytest.fixture(autouse=True)
def _no_ambient_azure_env(monkeypatch):
    # main.py's load_dotenv() can copy the real .env into os.environ during a
    # session, and Settings reads os.environ even with _env_file=None.
    for name in ("HARNESS_AZURE_API_KEY", "HARNESS_AZURE_ENDPOINT", "HARNESS_GUARD_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides) -> Settings:
    return Settings(**{"_env_file": None, "api_key": "test-key", **overrides})


def _context() -> GuardContext:
    return GuardContext(message="hi", system_prompt="You are a site guide.")


class _Foundry:
    """The Foundry endpoint behind build_input_guard's HTTP client."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.client_kwargs: dict = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "microsoft-decision-1",
                "answers": {id_: {"noul": 0.0, "type": "noul"} for id_ in CHECK_IDS},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )


@pytest.fixture
def foundry(monkeypatch) -> _Foundry:
    """Route the guard's httpx client to a fake endpoint, keeping the real constructor."""
    endpoint = _Foundry()
    real_client = httpx.AsyncClient

    class _Client(real_client):
        def __init__(self, **kwargs) -> None:
            endpoint.client_kwargs = dict(kwargs)
            super().__init__(transport=httpx.MockTransport(endpoint.handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return endpoint


def test_guard_off_builds_limits_only():
    guard = build_input_guard(_settings(
        input_guard_enabled=False, max_message_chars=10, max_thread_turns=3,
        daily_token_budget=500,
    ))
    assert guard.classifier_enabled is False
    assert guard.limits == GuardLimits(
        max_message_chars=10, max_thread_turns=3, daily_token_budget=500
    )


async def test_guard_on_with_a_resource_key_calls_the_decision_endpoint_tight(foundry):
    guard = build_input_guard(_settings(
        input_guard_enabled=True, azure_endpoint=ENDPOINT, azure_api_key="resource-key",
        guard_deployment="my-decision", guard_timeout_seconds=2.5,
    ))
    assert guard.classifier_enabled is True

    decision = await guard.screen(_context())

    assert (decision.allowed, decision.stage) == (True, "classifier")
    [request] = foundry.requests
    assert str(request.url) == f"{ENDPOINT}/providers/microsoft/v1/systemone"
    assert request.headers["api-key"] == "resource-key"
    body = json.loads(request.content)
    assert body["model"] == "my-decision"
    assert body["state"]["visitor_message"] == "hi"
    assert set(body["questions"]) == set(CHECK_IDS)
    assert foundry.client_kwargs["timeout"] == 2.5
    await guard.aclose()


async def test_guard_on_without_a_key_authenticates_as_the_managed_identity(foundry, monkeypatch):
    asked: list[tuple[str, ...]] = []

    class _Credential:
        def get_token(self, *scopes: str, **kwargs) -> AccessToken:
            asked.append(scopes)
            return AccessToken("entra-token", int(time.time()) + 3600)

    monkeypatch.setattr("azure.identity.DefaultAzureCredential", _Credential)
    guard = build_input_guard(_settings(input_guard_enabled=True, azure_endpoint=ENDPOINT))

    decision = await guard.screen(_context())

    assert (decision.allowed, decision.stage) == (True, "classifier")
    [request] = foundry.requests
    assert request.headers["authorization"] == "Bearer entra-token"
    assert "api-key" not in request.headers
    assert asked == [("https://cognitiveservices.azure.com/.default",)]
    await guard.aclose()
