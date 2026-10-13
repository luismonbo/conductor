"""Tools for the website's public assistant: document search over its own collections."""
from __future__ import annotations

from harness.adapters.tools.search_documents import SearchDocumentsTool
from harness.agents.definition import ToolDeps
from harness.core.tools.registry import ToolRegistry

# The only collections this agent may search, whatever else the deployment has
# ingested. search_documents refuses any other collection a model names.
COLLECTIONS = ("docs",)


def build_tools(deps: ToolDeps) -> ToolRegistry:
    """Document search is required: answering from those documents is the
    website assistant's whole job, so it does not start without it."""
    if deps.retriever is None or deps.vector_store is None:
        raise RuntimeError(
            "the website agent needs document search, but the RAG backend is unavailable"
        ) from deps.rag_error
    registry = ToolRegistry()
    registry.register(SearchDocumentsTool(
        deps.retriever, deps.vector_store, COLLECTIONS[0], deps.settings.rag_k, list(COLLECTIONS),
    ))
    return registry
