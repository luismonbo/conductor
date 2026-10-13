"""The calculator explains a bad expression to the model instead of failing opaquely."""
from __future__ import annotations

import re

import pytest

from harness.adapters.tools.calculator import CalculatorTool
from harness.core.tools.registry import ToolRegistry
from harness.core.types import ToolCall


async def _dispatch(arguments: dict):
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return await registry.dispatch(ToolCall(id="c1", name="calculator", arguments=arguments))


async def test_evaluates_an_expression():
    result = await _dispatch({"expression": "3 * (4 + 2)"})

    assert not result.is_error
    assert result.content == "18"


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("1 / 0", "division by zero"),
        ("2 +", "invalid syntax"),
        ("abs(-1)", "unsupported expression"),
        ("'a' * 3", "non-numeric constant"),
        # The C library words a float overflow: macOS, then glibc (CI).
        ("2.0 ** 10000", "Result too large|Numerical result out of range"),
    ],
)
async def test_a_bad_expression_is_explained_to_the_model(expression, reason):
    result = await _dispatch({"expression": expression})

    assert result.is_error
    assert result.content.startswith(f"Tool 'calculator' failed: cannot evaluate {expression!r}")
    assert re.search(reason, result.content)


@pytest.mark.parametrize("arguments", [{}, {"expression": ""}, {"expression": None}], ids=repr)
async def test_a_missing_expression_is_explained_to_the_model(arguments):
    result = await _dispatch(arguments)

    assert result.is_error
    assert "expression is required" in result.content
