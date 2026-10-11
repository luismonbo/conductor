"""Fixture: an agent that cannot work without document search."""
from harness.agents.definition import AgentDefinition, ToolDeps
from harness.core.tools.registry import ToolRegistry


def _tools(deps: ToolDeps) -> ToolRegistry:
    if deps.retriever is None:
        raise RuntimeError("needs_rag requires the RAG backend") from deps.rag_error
    return ToolRegistry()


AGENT = AgentDefinition(name="needs_rag", prompt="You need documents.", tools=_tools)
