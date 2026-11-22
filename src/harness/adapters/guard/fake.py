"""Scripted GuardClassifier for tests: fixed probabilities, or a raised error."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from harness.core.guard import ClassifierResult, GuardQuestion


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
        self.calls: list[dict[str, str]] = []  # the state of each call
        self.asked: list[tuple[str, ...]] = []  # the question ids of each call, in order
        self.closed = False

    async def classify(
        self, state: Mapping[str, str], questions: Sequence[GuardQuestion]
    ) -> ClassifierResult:
        self.calls.append(dict(state))
        self.asked.append(tuple(q.id for q in questions))
        if self._error is not None:
            raise self._error
        return ClassifierResult(
            probabilities={q.id: self._probabilities.get(q.id, 0.0) for q in questions},
            model=self._model,
            request_id="fake-request",
            input_tokens=0,
        )

    async def aclose(self) -> None:
        self.closed = True
