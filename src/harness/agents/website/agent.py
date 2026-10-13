"""The website's public assistant."""
from harness.agents.definition import AgentDefinition
from harness.agents.website.prompt import SYSTEM_PROMPT
from harness.agents.website.tools import build_tools

AGENT = AgentDefinition(name="website", prompt=SYSTEM_PROMPT, tools=build_tools)
