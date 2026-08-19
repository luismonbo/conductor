"""Tool registry for the default agent."""
from __future__ import annotations

from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.recall import RecallTool
from harness.adapters.tools.remember import RememberTool
from harness.adapters.tools.search_documents import SearchDocumentsTool
from harness.core.memory.store import LongTermMemory
from harness.core.rag.ports import Retriever, VectorStore
from harness.core.tools.registry import ToolRegistry


def build_registry(
    long_term: LongTermMemory | None = None,
    retriever: Retriever | None = None,
    vector_store: VectorStore | None = None,
    default_collection: str = "papers",
    default_k: int = 5,
    collections: list[str] | None = None,
) -> ToolRegistry:
    if long_term is None:
        long_term = InMemoryLongTerm()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RecallTool(long_term))
    registry.register(RememberTool(long_term))
    if retriever is not None and vector_store is not None:
        registry.register(SearchDocumentsTool(
            retriever, vector_store, default_collection, default_k, collections,
        ))
    return registry
