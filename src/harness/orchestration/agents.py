"""Load an agent's definition by name.

An agent is a folder harness/agents/<name>/ whose agent.py defines AGENT, an
AgentDefinition. Loading by name means there is no registry to edit, a
gitignored (private) agent folder loads exactly like a committed one, and only
the agents a deployment names are ever imported. See
docs/superpowers/specs/2026-10-08-agent-architecture-design.md.
"""
from __future__ import annotations

import importlib
import re

from harness.agents.definition import AgentDefinition

AGENTS_PACKAGE = "harness.agents"
_NAME = re.compile(r"[a-z][a-z0-9_]*")


class AgentLoadError(ValueError):
    """An agent name that does not resolve to a valid agent folder."""


def load_agent(name: str, package: str = AGENTS_PACKAGE) -> AgentDefinition:
    """The AgentDefinition in <package>/<name>/agent.py.

    Raises AgentLoadError for an invalid name, a missing folder or agent.py, a
    missing or wrong-type AGENT, or a name mismatch. An error raised by the
    agent's own code propagates unchanged: a typo inside an agent must not
    read as "no such agent".
    """
    if not _NAME.fullmatch(name):
        raise AgentLoadError(
            f"agent name {name!r} is invalid: use lowercase letters, digits and "
            "underscores, starting with a letter (a folder under src/harness/agents/)"
        )
    module_name = f"{package}.{name}.agent"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in (f"{package}.{name}", module_name):
            raise
        expected = module_name.replace(".", "/") + ".py"
        raise AgentLoadError(f"no agent named {name!r}: expected {expected}") from exc
    definition = getattr(module, "AGENT", None)
    if not isinstance(definition, AgentDefinition):
        raise AgentLoadError(f"{module_name} must define AGENT = AgentDefinition(...)")
    if definition.name != name:
        raise AgentLoadError(
            f"{module_name} defines an agent named {definition.name!r}; "
            f"its name must match its folder, {name!r}"
        )
    return definition
