"""Scripted fake LLMClient.

Lets the entire vertical slice run with zero credentials and makes the agent
loop deterministically testable. You hand it a queue of LLMResponses; each
generate() or stream() call pops the next one. This is also the 'fake adapter'
used by tests/unit so core logic is tested with no network.
"""
from __future__ import annotations

from collections import deque
from typing import AsyncGenerator

from harness.core.llm.client import LLMClient
from harness.core.types import LLMResponse, Message, ToolSpec


class FakeLLMClient(LLMClient):
    def __init__(self, scripted: list[LLMResponse], repeat_last: bool = False) -> None:
        self._queue: deque[LLMResponse] = deque(scripted)
        self._repeat_last = repeat_last
        self._last: LLMResponse | None = None
        self.calls: list[list[Message]] = []
        self.requested_models: list[str | None] = []

    def _next_response(self) -> LLMResponse:
        if not self._queue:
            if self._repeat_last and self._last is not None:
                return self._last
            return LLMResponse(text="(no scripted response left)")
        self._last = self._queue.popleft()
        return self._last

    @property
    def model_id(self) -> str:
        return "fake-llm"

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.requested_models.append(model)
        self.calls.append(list(messages))
        return self._next_response()

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str | LLMResponse, None]:
        self.requested_models.append(model)
        self.calls.append(list(messages))
        response = self._next_response()

        # Yield text word-by-word so tests can assert on individual token events.
        if response.text:
            words = response.text.split()
            for i, word in enumerate(words):
                suffix = " " if i < len(words) - 1 else ""
                yield word + suffix

        yield response
