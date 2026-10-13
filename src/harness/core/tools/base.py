"""Tool protocol. Each concrete tool implements this in adapters/tools/.

A tool exposes a JSON-schema spec (handed to the model) and an async run().
The agent never references a tool by name in code — it asks the ToolRegistry
for specs and dispatches results by id. Adding a tool is adding a file plus a
registration call; the agent loop is never edited.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from harness.core.types import ToolSpec


class ToolError(Exception):
    """A tool failure whose message the model may read.

    Raise it for something the model can act on: a missing argument, a
    collection it may not search. Any other exception from run() reaches the
    model only as "Tool '<name>' failed." (ToolRegistry.dispatch): its text may
    carry internals (a provider's error body, a path) the model could repeat
    to a visitor.
    """


@runtime_checkable
class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    @property
    def requires_approval(self) -> bool:
        """True for mutating tools that require human approval before execution."""
        return False

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute and return a string result. Raise ToolError for a failure the
        model should read; the registry masks any other exception, so one bad
        tool can't crash the agent loop or leak its internals."""
        ...
