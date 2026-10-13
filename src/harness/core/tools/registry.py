"""ToolRegistry: the mechanism that makes 'add a tool without touching the
agent' literally true.

The agent asks the registry for specs() to show the model, then calls
dispatch() with whatever the model requested. Unknown tools and tool
exceptions become error ToolResults rather than crashing the loop — important
with small models that hallucinate tool names. Only a ToolError's message
reaches the model.
"""
from __future__ import annotations

import logging

from harness.core.tools.base import Tool, ToolError
from harness.core.types import ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(
                    f"Unknown tool '{call.name}'. "
                    f"Available tools: {', '.join(self.names()) or 'none'}."
                ),
                is_error=True,
            )
        try:
            content = await tool.run(call.arguments)
            return ToolResult(call.id, call.name, content, is_error=False)
        except ToolError as exc:
            # A failure the tool explained for the model: its message is meant to be read.
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool '{call.name}' failed: {exc}",
                is_error=True,
            )
        except Exception:  # noqa: BLE001 - deliberately broad at boundary
            # Anything else may carry internals (a provider's error body, a path)
            # the model could repeat to a visitor: the detail goes to the log only.
            logger.exception("tool %s failed", call.name)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool '{call.name}' failed.",
                is_error=True,
            )
