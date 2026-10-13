"""Only a ToolError's message reaches the model; anything else is masked and logged."""
from __future__ import annotations

import logging
from typing import Any

from harness.core.tools.base import ToolError
from harness.core.tools.registry import ToolRegistry
from harness.core.types import ToolCall, ToolSpec


class _Failing:
    """A tool whose run() raises the exception it was given."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="lookup", description="test tool",
            parameters={"type": "object", "properties": {}},
        )

    @property
    def requires_approval(self) -> bool:
        return False

    async def run(self, arguments: dict[str, Any]) -> str:
        raise self._exc


async def _dispatch(exc: Exception):
    registry = ToolRegistry()
    registry.register(_Failing(exc))
    return await registry.dispatch(ToolCall(id="c1", name="lookup", arguments={}))


async def test_a_tool_error_message_reaches_the_model():
    result = await _dispatch(ToolError("collection 'x' is not available"))

    assert result.is_error
    assert result.content == "Tool 'lookup' failed: collection 'x' is not available"


async def test_any_other_exception_is_masked_and_logged(caplog):
    leak = RuntimeError("Error code: 401 - {'error': {'code': 'PermissionDenied'}} at /app/data")

    with caplog.at_level(logging.ERROR, logger="harness.core.tools.registry"):
        result = await _dispatch(leak)

    assert result.is_error
    assert result.content == "Tool 'lookup' failed."
    [record] = [r for r in caplog.records if r.name == "harness.core.tools.registry"]
    assert record.exc_info is not None and record.exc_info[1] is leak
    assert "lookup" in record.getMessage()


async def test_an_unknown_tool_still_lists_the_available_ones():
    registry = ToolRegistry()
    registry.register(_Failing(ValueError()))

    result = await registry.dispatch(ToolCall(id="c1", name="nope", arguments={}))

    assert result.is_error
    assert result.content == "Unknown tool 'nope'. Available tools: lookup."
