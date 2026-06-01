"""Tool protocol. Each concrete tool implements this in adapters/tools/.

A tool exposes a JSON-schema spec (handed to the model) and an async run().
The agent never references a tool by name in code — it asks the ToolRegistry
for specs and dispatches results by id. Adding a tool is adding a file plus a
registration call; the agent loop is never edited.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from harness.core.types import ToolSpec


@runtime_checkable
class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    @property
    def requires_approval(self) -> bool:
        """True for mutating tools that require human approval before execution."""
        return False

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute and return a string result. Raise on hard failure; the
        registry wraps exceptions into an error ToolResult so one bad tool
        can't crash the agent loop."""
        ...
