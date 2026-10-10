"""The default agent (Lumina): a general assistant with the calculator,
optional shared memory and document search."""
from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.agents.default.tools import build_tools
from harness.agents.definition import AgentDefinition

AGENT = AgentDefinition(name="default", prompt=SYSTEM_PROMPT, tools=build_tools)
