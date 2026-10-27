"""AzureDecisionClassifier: checks become noul questions, answers become probabilities, failures are classified."""
from __future__ import annotations

import json

import httpx
import pytest

from harness.adapters.guard.azure_decision import AzureDecisionClassifier
from harness.core.guard import Criterion, GuardCheck, GuardClassifierError

ENDPOINT = "https://foundry.example"
URL = "https://foundry.example/providers/microsoft/v1/systemone"

CHECK_A = GuardCheck(
    id="a",
    question="Is `visitor_message` A?",
    yes=Criterion("yes-def", ("y1", "y2")),
    no=Criterion("no-def", ("n1",)),
    threshold=0.7,
    refusal="r",
)
CHECK_B = GuardCheck(
    id="b", question="Is it B?", yes=Criterion("y"), no=Criterion("n"),
    threshold=0.7, refusal="r",
)


class _Service:
    """The Foundry endpoint: hands out the queued replies in order, records each request.

    A request beyond the queued replies raises IndexError inside the transport, so
    a retry the test did not expect fails loudly instead of passing quietly.
    """

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.requests: list[httpx.Request] = []
        self._replies = list(replies)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]


def _classifier(service: _Service, endpoint: str = ENDPOINT, **auth) -> AzureDecisionClassifier:
    client = httpx.AsyncClient(transport=httpx.MockTransport(service))
    return AzureDecisionClassifier(
        client, endpoint, "Decision-1", **(auth or {"api_key": "k"})
    )


def _answers(**noul: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "microsoft-decision-1",
            "answers": {key: {"noul": value, "type": "noul"} for key, value in noul.items()},
            "usage": {"input_tokens": 321, "output_tokens": 2},
        },
        headers={"x-request-id": "req-1"},
    )


def _error(status: int, code: str = "oops") -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": "text"}})


async def test_posts_the_deployment_state_and_one_noul_question_per_check():
    service = _Service(_answers(a=0.1, b=0.2))
    await _classifier(service).classify({"visitor_message": "hi"}, [CHECK_A, CHECK_B])

    assert len(service.requests) == 1
    assert (service.requests[0].method, str(service.requests[0].url)) == ("POST", URL)
    body = service.bodies[0]
    assert body["model"] == "Decision-1"
    assert body["state"] == {"visitor_message": "hi"}
    assert set(body["questions"]) == {"a", "b"}
    question = body["questions"]["a"]
    assert question["type"] == "noul"
    assert question["instructions"] == "Is `visitor_message` A?"
    assert set(question["criteria"]) == {"true", "false"}


async def test_each_side_of_a_check_reaches_the_service_with_its_own_examples():
    service = _Service(_answers(a=0.1, b=0.1))
    await _classifier(service).classify({}, [CHECK_A, CHECK_B])

    criteria = service.bodies[0]["questions"]["a"]["criteria"]
    assert criteria["true"].startswith("yes-def")
    assert "y1" in criteria["true"] and "y2" in criteria["true"]
    assert "n1" not in criteria["true"]
    assert criteria["false"].startswith("no-def")
    assert "n1" in criteria["false"]
    assert "y1" not in criteria["false"]
    # A side without examples is just its definition, with no empty examples block.
    assert service.bodies[0]["questions"]["b"]["criteria"] == {"true": "y", "false": "n"}


async def test_a_trailing_slash_on_the_endpoint_does_not_change_the_url():
    service = _Service(_answers(a=0.1))
    await _classifier(service, endpoint=ENDPOINT + "/").classify({}, [CHECK_A])
    assert str(service.requests[0].url) == URL


async def test_maps_answers_and_metadata():
    service = _Service(_answers(a=0.91, b=0.05))
    result = await _classifier(service).classify({}, [CHECK_A, CHECK_B])

    assert result.probabilities == {"a": 0.91, "b": 0.05}
    assert result.model == "microsoft-decision-1"
    assert result.request_id == "req-1"
    assert result.input_tokens == 321


async def test_a_missing_request_id_header_is_none_not_an_error():
    reply = httpx.Response(
        200,
        json={"model": "m", "answers": {"a": {"noul": 0.1, "type": "noul"}},
              "usage": {"input_tokens": 1, "output_tokens": 1}},
    )
    result = await _classifier(_Service(reply)).classify({}, [CHECK_A])
    assert result.request_id is None


async def test_authenticates_with_the_api_key_header():
    service = _Service(_answers(a=0.1))
    await _classifier(service, api_key="resource-key").classify({}, [CHECK_A])

    assert service.requests[0].headers["api-key"] == "resource-key"
    assert "authorization" not in service.requests[0].headers


async def test_authenticates_with_a_bearer_token_from_the_provider():
    service = _Service(_answers(a=0.1))
    await _classifier(service, token_provider=lambda: "entra-token").classify({}, [CHECK_A])

    assert service.requests[0].headers["authorization"] == "Bearer entra-token"
    assert "api-key" not in service.requests[0].headers


async def test_a_token_provider_failure_is_misconfigured_and_sends_nothing():
    def no_credential() -> str:
        raise RuntimeError("no credential available")

    service = _Service()
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service, token_provider=no_credential).classify({}, [CHECK_A])

    assert info.value.cause == "RuntimeError"
    assert info.value.misconfigured is True
    assert service.requests == []


@pytest.mark.parametrize(
    "auth",
    [{}, {"api_key": "k", "token_provider": lambda: "t"}],
    ids=["neither", "both"],
)
async def test_exactly_one_credential_is_required(auth):
    client = httpx.AsyncClient()
    try:
        with pytest.raises(ValueError, match="exactly one"):
            AzureDecisionClassifier(client, ENDPOINT, "Decision-1", **auth)
    finally:
        await client.aclose()


async def test_a_check_the_service_did_not_answer_is_misconfigured():
    service = _Service(_answers(a=0.1))
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({}, [CHECK_A, CHECK_B])

    assert info.value.cause == "MissingAnswer"
    assert info.value.misconfigured is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_rejections_that_recur_on_every_call_are_misconfigured_and_not_retried(status):
    service = _Service(_error(status))
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({}, [CHECK_A])

    assert info.value.cause == f"HTTP{status}"
    assert info.value.misconfigured is True
    assert len(service.requests) == 1


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
async def test_a_transient_status_is_retried_once(status):
    service = _Service(_error(status), _answers(a=0.4))
    result = await _classifier(service).classify({}, [CHECK_A])

    assert result.probabilities == {"a": 0.4}
    assert len(service.requests) == 2


@pytest.mark.parametrize("status", [408, 500, 503])
async def test_a_transient_status_that_repeats_is_not_misconfigured(status):
    service = _Service(_error(status), _error(status))
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({}, [CHECK_A])

    assert info.value.cause == f"HTTP{status}"
    assert info.value.misconfigured is False
    assert len(service.requests) == 2  # one retry, never a third


async def test_a_rate_limit_is_transient_and_not_retried():
    service = _Service(_error(429))
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({}, [CHECK_A])

    assert info.value.cause == "HTTP429"
    assert info.value.misconfigured is False
    assert len(service.requests) == 1


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
    ids=["connect", "read-timeout"],
)
async def test_a_network_failure_that_repeats_is_transient(error):
    service = _Service(error, error)
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({}, [CHECK_A])

    assert info.value.cause == type(error).__name__
    assert info.value.misconfigured is False
    assert len(service.requests) == 2


async def test_a_network_failure_that_clears_up_is_retried_to_success():
    service = _Service(httpx.ConnectError("refused"), _answers(a=0.4))
    result = await _classifier(service).classify({}, [CHECK_A])

    assert result.probabilities == {"a": 0.4}
    assert len(service.requests) == 2


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"model": "m"}),
        httpx.Response(200, json={"answers": {"a": {"type": "noul"}}}),
        httpx.Response(200, json={"answers": {"a": {"noul": "high", "type": "noul"}}}),
        httpx.Response(200, json={"answers": {"a": {"noul": 1.5, "type": "noul"}}}),
        httpx.Response(200, json={"answers": {"a": {"noul": True, "type": "noul"}}}),
    ],
    ids=["not-json", "not-an-object", "no-answers", "no-probability", "string-probability",
         "out-of-range", "boolean"],
)
async def test_a_response_that_does_not_fit_the_api_is_misconfigured(reply):
    service = _Service(reply)
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({}, [CHECK_A])

    assert info.value.cause == "InvalidResponse"
    assert info.value.misconfigured is True
    assert len(service.requests) == 1


async def test_an_error_names_the_service_code_and_never_the_visitor_message():
    # The service's free text can echo the request, and this message reaches the logs.
    echoing = httpx.Response(
        400,
        json={"error": {"code": "unsupported_request_argument",
                        "message": "cannot parse: my private question"}},
    )
    service = _Service(echoing)
    with pytest.raises(GuardClassifierError) as info:
        await _classifier(service).classify({"visitor_message": "my private question"}, [CHECK_A])

    assert "unsupported_request_argument" in str(info.value)
    assert "my private question" not in str(info.value)


async def test_aclose_closes_the_http_client():
    client = httpx.AsyncClient(transport=httpx.MockTransport(_Service()))
    await AzureDecisionClassifier(client, ENDPOINT, "Decision-1", api_key="k").aclose()
    assert client.is_closed
