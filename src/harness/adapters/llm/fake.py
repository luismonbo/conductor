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
    def __init__(self, scripted: list[LLMResponse]) -> None:
        self._queue: deque[LLMResponse] = deque(scripted)
        self.calls: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return "fake-llm"

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._queue:
            return LLMResponse(text="(no scripted response left)")
        return self._queue.popleft()

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[str | LLMResponse, None]:
        self.calls.append(list(messages))
        if not self._queue:
            response = LLMResponse(text="(no scripted response left)")
        else:
            response = self._queue.popleft()

        # Yield text word-by-word so tests can assert on individual token events.
        if response.text:
            words = response.text.split()
            for i, word in enumerate(words):
                suffix = " " if i < len(words) - 1 else ""
                yield word + suffix

        yield response
