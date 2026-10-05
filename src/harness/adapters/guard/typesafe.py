"""TypeSafe (Jev) adapter for the input guard's GuardClassifier port.

One system_one request per message: every GuardCheck becomes a Noul whose
criteria carry a structured {definition, examples} object on each side, which
TypeSafe recommends when the yes/no boundary is subtle. The caller pins the
model (Settings.guard_model): thresholds are calibrated against one version,
and an alias like jev-latest can move underneath them.

Every SDK failure is re-raised as GuardClassifierError so the policy never
sees a vendor type. Never set TYPESAFE_LOG_LEVEL=debug in production: the SDK
then logs request bodies (visitor messages) unredacted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from typesafe_sdk import (
    Noul,
    NoulCriteria,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafePermissionDeniedError,
)

from harness.core.guard import (
    ClassifierResult,
    Criterion,
    GuardCheck,
    GuardClassifierError,
)


class TypeSafeClassifier:
    def __init__(self, client: Any, model: str) -> None:
        # `client` is an AsyncTypeSafeClient; typed Any so tests can pass a stub.
        self._client = client
        self._model = model

    async def classify(
        self, state: Mapping[str, str], checks: Sequence[GuardCheck]
    ) -> ClassifierResult:
        questions = {check.id: _noul(check) for check in checks}
        try:
            response = await self._client.system_one(
                dict(state), questions, model=self._model
            )
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as exc:
            raise GuardClassifierError(
                str(exc), cause=type(exc).__name__, misconfigured=True
            ) from exc
        except TypeSafeError as exc:
            raise GuardClassifierError(str(exc), cause=type(exc).__name__) from exc

        probabilities: dict[str, float] = {}
        for check in checks:
            answer = response.nouls.get(check.id)
            if answer is None:
                raise GuardClassifierError(
                    f"no answer for check {check.id!r}", cause="MissingAnswer"
                )
            probabilities[check.id] = float(answer.noul)
        return ClassifierResult(
            probabilities=probabilities,
            model=response.model,
            request_id=_request_id(response),
            input_tokens=response.usage.input_tokens,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _noul(check: GuardCheck) -> Noul:
    return Noul(
        instructions=check.question,
        criteria=NoulCriteria(true=_criterion(check.yes), false=_criterion(check.no)),
    )


def _criterion(criterion: Criterion) -> dict[str, Any]:
    return {"definition": criterion.definition, "examples": list(criterion.examples)}


def _request_id(response: Any) -> str | None:
    # The SDK's request_id property raises (rather than returning None) when
    # the x-typesafe-request-id header is absent.
    try:
        return response.request_id
    except Exception:
        return None
