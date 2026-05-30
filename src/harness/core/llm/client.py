"""LLM client and tool-call parsing protocols.

This is the seam that lets you develop against Azure OpenAI and later swap in
Gemma-4 on a Pi without touching the agent loop. The agent depends only on
these Protocols; concrete clients live in adapters/llm/.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.core.types import LLMResponse, Message, ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """Generates one completion given a transcript and the available tools.

    Implementations MUST return tool calls via LLMResponse.tool_calls regardless
    of whether the underlying model emits them natively or in prose — that
    normalization is the ToolCallParser's job, wired inside the adapter.
    """

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...

    @property
    def model_id(self) -> str: ...
