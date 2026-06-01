"""LLM client and tool-call parsing protocols.

This is the seam that lets you develop against Azure OpenAI and later swap in
Gemma-4 on a Pi without touching the agent loop. The agent depends only on
these Protocols; concrete clients live in adapters/llm/.
"""
from __future__ import annotations

from typing import AsyncGenerator, Protocol, runtime_checkable

from harness.core.types import LLMResponse, Message, ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """Generates completions given a transcript and the available tools.

    Two call styles:
      generate() — blocking, returns one LLMResponse. Used by the legacy
                   /chat endpoint and as a convenience in tests.
      stream()   — async generator; yields str tokens as they arrive, then
                   yields a single LLMResponse as the final item carrying
                   tool_calls, usage, and finish_reason.
    """

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[str | LLMResponse, None]:
        """Implement as an async generator function (async def + yield).

        Callers: async for item in client.stream(messages, tools):
        """
        ...

    @property
    def model_id(self) -> str: ...
