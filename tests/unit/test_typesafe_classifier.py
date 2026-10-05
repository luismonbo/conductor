"""TypeSafeClassifier: checks become Nouls, answers become probabilities, SDK errors are wrapped."""
from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from harness.adapters.guard.typesafe import TypeSafeClassifier
from harness.core.guard import Criterion, GuardCheck, GuardClassifierError

CHECK_A = GuardCheck(
    id="a",
    question="Is `visitor_message` A?",
    yes=Criterion("yes-def", ("y1",)),
    no=Criterion("no-def", ("n1", "n2")),
    threshold=0.7,
    refusal="r",
)
CHECK_B = GuardCheck(
    id="b", question="Is it B?", yes=Criterion("y"), no=Criterion("n"),
    threshold=0.7, refusal="r",
)


class _StubClient:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []
        self.closed = False

    async def system_one(self, state, questions, *, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def _response(nouls: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        model="jev-1.13.0",
        nouls={k: SimpleNamespace(noul=v) for k, v in nouls.items()},
        usage=SimpleNamespace(input_tokens=321, output_tokens=0),
        request_id="req-1",
    )


def _api_error(cls, status: int):
    return cls(status=status, body=None, headers=httpx2.Headers())


async def test_sends_one_request_with_a_noul_per_check_and_the_pinned_model():
    client = _StubClient(_response({"a": 0.1, "b": 0.2}))
    await TypeSafeClassifier(client, model="jev-1.13.0").classify(
        {"visitor_message": "hi"}, [CHECK_A, CHECK_B]
    )
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "jev-1.13.0"
    assert call["state"] == {"visitor_message": "hi"}
    assert set(call["questions"]) == {"a", "b"}
    assert call["questions"]["a"].model_dump() == {
        "type": "noul",
        "instructions": "Is `visitor_message` A?",
        "criteria": {
            "true": {"definition": "yes-def", "examples": ["y1"]},
            "false": {"definition": "no-def", "examples": ["n1", "n2"]},
        },
    }


async def test_maps_answers_and_metadata():
    result = await TypeSafeClassifier(
        _StubClient(_response({"a": 0.91, "b": 0.05})), model="m"
    ).classify({}, [CHECK_A, CHECK_B])
    assert result.probabilities == {"a": 0.91, "b": 0.05}
    assert result.model == "jev-1.13.0"
    assert result.request_id == "req-1"
    assert result.input_tokens == 321


async def test_a_missing_request_id_is_none_not_an_error():
    class _NoRequestId:
        model = "jev-1.13.0"
        nouls = {"a": SimpleNamespace(noul=0.1)}
        usage = SimpleNamespace(input_tokens=1, output_tokens=0)

        @property
        def request_id(self):
            raise RuntimeError("The response did not include a request ID.")

    result = await TypeSafeClassifier(_StubClient(_NoRequestId()), model="m").classify(
        {}, [CHECK_A]
    )
    assert result.request_id is None


async def test_a_missing_answer_raises():
    with pytest.raises(GuardClassifierError) as info:
        await TypeSafeClassifier(_StubClient(_response({"a": 0.1})), model="m").classify(
            {}, [CHECK_A, CHECK_B]
        )
    assert info.value.cause == "MissingAnswer"
    assert info.value.misconfigured is True


@pytest.mark.parametrize(
    "error",
    [
        _api_error(TypeSafeAuthenticationError, 401),
        _api_error(TypeSafePermissionDeniedError, 403),
        _api_error(TypeSafeBadRequestError, 400),
        _api_error(TypeSafeNotFoundError, 404),  # e.g. a retired pinned model
        _api_error(TypeSafeUnprocessableEntityError, 422),
        TypeSafeAPIResponseValidationError(
            status=200, body=None, headers=httpx2.Headers(), field_path="nouls"
        ),
    ],
)
async def test_failures_that_recur_on_every_call_are_misconfigured(error):
    with pytest.raises(GuardClassifierError) as info:
        await TypeSafeClassifier(_StubClient(error=error), model="m").classify({}, [CHECK_A])
    assert info.value.cause == type(error).__name__
    assert info.value.misconfigured is True


@pytest.mark.parametrize(
    "error",
    [
        _api_error(TypeSafeRateLimitError, 429),
        _api_error(TypeSafeInternalServerError, 503),
        _api_error(TypeSafeInternalServerError, 529),  # overloaded
        _api_error(TypeSafeAPIError, 408),
        TypeSafeAPIConnectionError("unreachable"),
        TypeSafeAPITimeoutError(timeout=3.0),
    ],
)
async def test_transient_errors_are_wrapped_as_not_misconfigured(error):
    with pytest.raises(GuardClassifierError) as info:
        await TypeSafeClassifier(_StubClient(error=error), model="m").classify({}, [CHECK_A])
    assert info.value.cause == type(error).__name__
    assert info.value.misconfigured is False


async def test_aclose_closes_the_sdk_client():
    client = _StubClient()
    await TypeSafeClassifier(client, model="m").aclose()
    assert client.closed
