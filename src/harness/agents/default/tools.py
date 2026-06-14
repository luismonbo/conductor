"""Tool registry for the default agent."""
from __future__ import annotations

from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.recall import RecallTool
from harness.adapters.tools.remember import RememberTool
from harness.core.memory.store import LongTermMemory
from harness.core.tools.registry import ToolRegistry


def build_registry(long_term: LongTermMemory | None = None) -> ToolRegistry:
    if long_term is None:
        long_term = InMemoryLongTerm()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RecallTool(long_term))
    registry.register(RememberTool(long_term))
    return registry
