"""Fixture: the definition's name does not match its folder."""
from harness.agents.definition import AgentDefinition
from harness.core.tools.registry import ToolRegistry

AGENT = AgentDefinition(name="someone_else", prompt="x", tools=lambda deps: ToolRegistry())
