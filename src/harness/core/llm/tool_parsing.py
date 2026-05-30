"""Tool-call parsing strategy.

Two implementations live in adapters/llm/parsers.py:
  - NativeToolCallParser: reads structured tool_calls from a capable model
    (Azure OpenAI, hosted Gemma-4 26B/31B).
  - PromptedToolCallParser: injects a strict format into the prompt and
    tolerantly extracts calls from text — for small models (E2B on a Pi)
    that can't be trusted to emit structured calls reliably.

The agent loop never knows which is active; it's selected per backend in config.
Both must pass tests/contract/test_tool_call_parser.py against the same fixtures.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.core.types import ToolCall, ToolSpec


@runtime_checkable
class ToolCallParser(Protocol):
    def system_prompt_addendum(self, tools: list[ToolSpec]) -> str:
        """Extra instructions to inject. Native returns ""; prompted returns the
        format spec the small model must follow."""
        ...

    def extract(self, raw_text: str) -> tuple[str, list[ToolCall]]:
        """Split a raw model output into (assistant_text, tool_calls)."""
        ...
