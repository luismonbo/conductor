"""Tool protocol and registry unit tests."""
from __future__ import annotations

import pytest
from harness.adapters.tools.calculator import CalculatorTool
from harness.core.tools.registry import ToolRegistry
from harness.core.types import ToolCall


@pytest.mark.unit
def test_calculator_has_no_approval_by_default():
    tool = CalculatorTool()
    assert getattr(tool, "requires_approval", False) is False


@pytest.mark.unit
def test_registry_get_returns_tool():
    r = ToolRegistry()
    r.register(CalculatorTool())
    assert r.get("calculator") is not None


@pytest.mark.unit
def test_registry_get_returns_none_for_unknown():
    r = ToolRegistry()
    assert r.get("nonexistent") is None
