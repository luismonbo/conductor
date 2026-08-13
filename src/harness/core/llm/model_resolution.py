"""Model resolution for multi-model call sites.

Every LLM call site (an agent, or a tool that internally calls a model)
resolves its model by precedence: tool pin > agent pin > request override
(the UI picker) > global default. Pins beat the override on purpose: the
picker steers the conductor, never a component that was pinned for cost
or capability reasons. Names are LiteLLM profile names, defined once in
litellm_config.yaml — this module never sees a provider.
"""
from __future__ import annotations


def resolve_model(
    tool_pin: str | None = None,
    agent_pin: str | None = None,
    request_override: str | None = None,
    default: str | None = None,
) -> str | None:
    for candidate in (tool_pin, agent_pin, request_override, default):
        if candidate:
            return candidate
    return None
