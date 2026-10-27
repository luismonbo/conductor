"""Scripted GuardClassifier for tests: fixed probabilities, or a raised error."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from harness.core.guard import ClassifierResult, GuardCheck


class FakeClassifier:
    def __init__(
        self,
        probabilities: Mapping[str, float] | None = None,
        error: BaseException | None = None,
        model: str = "fake-decision",
    ) -> None:
        self._probabilities = dict(probabilities or {})
        self._error = error
        self._model = model
        self.calls: list[dict[str, str]] = []
        self.closed = False

    async def classify(
        self, state: Mapping[str, str], checks: Sequence[GuardCheck]
    ) -> ClassifierResult:
        self.calls.append(dict(state))
        if self._error is not None:
            raise self._error
        return ClassifierResult(
            probabilities={c.id: self._probabilities.get(c.id, 0.0) for c in checks},
            model=self._model,
            request_id="fake-request",
            input_tokens=0,
        )

    async def aclose(self) -> None:
        self.closed = True
