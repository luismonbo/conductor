"""What the model reads when it calls recall or remember with bad arguments.

Only a ToolError's message reaches the model (ToolRegistry.dispatch), so every
argument the model can get wrong is explained in one.
"""
from __future__ import annotations

import pytest

from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.recall import RecallTool
from harness.adapters.tools.remember import RememberTool
from harness.core.tools.base import Tool
from harness.core.tools.registry import ToolRegistry
from harness.core.types import ToolCall, ToolResult


async def _dispatch(tool: Tool, arguments: dict) -> ToolResult:
    """Run the tool the way the agent loop does, so a failure arrives as the model sees it."""
    registry = ToolRegistry()
    registry.register(tool)
    return await registry.dispatch(ToolCall(id="c1", name=tool.spec.name, arguments=arguments))


async def _memory_with(*facts: str) -> InMemoryLongTerm:
    memory = InMemoryLongTerm()
    for fact in facts:
        await memory.write(fact)
    return memory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", [{}, {"query": ""}, {"query": "   "}, {"query": None}], ids=repr,
)
async def test_recall_explains_a_missing_query(arguments):
    result = await _dispatch(RecallTool(await _memory_with("Luis lives in Madrid")), arguments)

    assert result.is_error
    assert "query is required" in result.content


@pytest.mark.asyncio
async def test_recall_reads_a_null_k_as_the_default():
    # Small models send null for an optional parameter.
    memory = await _memory_with("fact one", "fact two", "fact three", "fact four")

    result = await _dispatch(RecallTool(memory), {"query": "fact", "k": None})

    assert not result.is_error
    assert result.content.count("- (") == 3  # the schema's default


@pytest.mark.asyncio
@pytest.mark.parametrize("k", ["many", [3], float("inf"), float("nan")], ids=repr)
async def test_recall_explains_a_k_that_is_not_a_whole_number(k):
    result = await _dispatch(RecallTool(await _memory_with("fact one")), {"query": "fact", "k": k})

    assert result.is_error
    assert "k must be a whole number" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", [{}, {"text": ""}, {"text": "   "}, {"text": None}], ids=repr,
)
async def test_remember_explains_a_missing_fact_and_stores_nothing(arguments):
    memory = InMemoryLongTerm()

    result = await _dispatch(RememberTool(memory), arguments)

    assert result.is_error
    assert "text is required" in result.content
    assert await memory.search("anything") == []
