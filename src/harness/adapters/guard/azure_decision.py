"""Microsoft-Decision-1 adapter for the input guard's GuardClassifier port.

One request per message to the Foundry resource's decision endpoint
(POST {endpoint}/providers/microsoft/v1/systemone): every GuardCheck becomes a
`noul` question whose true/false criteria carry the check's definition and
examples as text, the shape Microsoft documents. `deployment` is the Foundry
deployment name, not the model name. The response names only the model family
(`microsoft-decision-1`), so the version is pinned on the deployment: thresholds
are calibrated against one version, and a deployment that auto-upgrades can move
underneath them.

Auth is exactly one of the resource's api key (local development) or a bearer
token provider (the Container App's managed identity), as in the LLM adapter.

Every failure is re-raised as GuardClassifierError so the policy never sees a
vendor type. Its message carries the HTTP status and the service's error code,
never the service's free text or the request: either can hold the visitor's
message, and this message reaches the logs.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from harness.core.guard import (
    ClassifierResult,
    Criterion,
    GuardCheck,
    GuardClassifierError,
)

_PATH = "/providers/microsoft/v1/systemone"
_ATTEMPTS = 2  # the call and one retry; InputGuard's own timeout caps the total


class AzureDecisionClassifier:
    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        deployment: str,
        *,
        api_key: str | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        if bool(api_key) == (token_provider is not None):
            raise ValueError("pass exactly one of api_key and token_provider")
        self._client = client
        self._url = endpoint.rstrip("/") + _PATH
        self._deployment = deployment
        self._api_key = api_key or ""
        self._token_provider = token_provider

    async def classify(
        self, state: Mapping[str, str], checks: Sequence[GuardCheck]
    ) -> ClassifierResult:
        body = {
            "model": self._deployment,
            "state": dict(state),
            "questions": {check.id: _question(check) for check in checks},
        }
        response = await self._post(body)
        if not response.is_success:
            raise _status_error(response)

        payload = _json_object(response)
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise _invalid("the response has no answers object")
        probabilities: dict[str, float] = {}
        for check in checks:
            if check.id not in answers:
                raise GuardClassifierError(
                    f"no answer for check {check.id!r}", cause="MissingAnswer",
                    misconfigured=True,
                )
            probabilities[check.id] = _probability(answers[check.id], check.id)
        return ClassifierResult(
            probabilities=probabilities,
            model=_text(payload.get("model")) or self._deployment,
            request_id=response.headers.get("x-request-id"),
            input_tokens=_input_tokens(payload),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        headers = await self._auth_headers()
        attempts_left = _ATTEMPTS
        while True:
            attempts_left -= 1
            try:
                response = await self._client.post(self._url, json=body, headers=headers)
            except httpx.TransportError as exc:  # connect, timeout, read, protocol
                if not attempts_left:
                    raise GuardClassifierError(
                        f"{type(exc).__name__} calling the decision endpoint",
                        cause=type(exc).__name__, misconfigured=False,
                    ) from exc
                continue
            if attempts_left and _is_retried(response.status_code):
                continue
            return response

    async def _auth_headers(self) -> dict[str, str]:
        if self._token_provider is None:
            return {"api-key": self._api_key}
        try:
            # The provider is sync (azure-identity) and refreshes over the network.
            token = await asyncio.to_thread(self._token_provider)
        except Exception as exc:
            raise GuardClassifierError(
                f"{type(exc).__name__} getting an Entra token for the decision endpoint",
                cause=type(exc).__name__, misconfigured=True,
            ) from exc
        return {"Authorization": f"Bearer {token}"}


def _question(check: GuardCheck) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": check.question,
        "criteria": {"true": _criterion(check.yes), "false": _criterion(check.no)},
    }


def _criterion(criterion: Criterion) -> str:
    if not criterion.examples:
        return criterion.definition
    examples = "\n".join(f"- {example}" for example in criterion.examples)
    return f"{criterion.definition}\nExamples:\n{examples}"


def _is_retried(status: int) -> bool:
    return status == 408 or status >= 500


def _is_transient(status: int) -> bool:
    """True for failures that clear up on their own.

    Everything else (a rejected credential or role, a retired or misspelled
    deployment, a request that no longer fits the API) recurs on every call
    until someone fixes it.
    """
    return status in (408, 429) or status >= 500


def _status_error(response: httpx.Response) -> GuardClassifierError:
    status = response.status_code
    code = _error_code(response)
    return GuardClassifierError(
        f"HTTP {status}" + (f" {code}" if code else ""),
        cause=f"HTTP{status}",
        misconfigured=not _is_transient(status),
    )


def _error_code(response: httpx.Response) -> str:
    try:
        error = response.json()["error"]
        code = error["code"]
    except (ValueError, KeyError, TypeError):
        return ""
    return code[:60] if isinstance(code, str) else ""


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise _invalid("the response is not JSON") from exc
    if not isinstance(payload, dict):
        raise _invalid("the response is not a JSON object")
    return payload


def _probability(answer: Any, check_id: str) -> float:
    value = answer.get("noul") if isinstance(answer, dict) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise _invalid(f"the answer for check {check_id!r} has no probability")
    return float(value)


def _input_tokens(payload: Mapping[str, Any]) -> int:
    usage = payload.get("usage")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _invalid(reason: str) -> GuardClassifierError:
    return GuardClassifierError(reason, cause="InvalidResponse", misconfigured=True)
