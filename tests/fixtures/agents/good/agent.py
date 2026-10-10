"""Fixture: a valid agent with one tool and a model pin."""
from harness.adapters.tools.calculator import CalculatorTool
from harness.agents.definition import AgentDefinition, ToolDeps
from harness.core.tools.registry import ToolRegistry


def _tools(deps: ToolDeps) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


AGENT = AgentDefinition(
    name="good",
    prompt="You are the good fixture agent.",
    tools=_tools,
    model="fixture-model",
)
