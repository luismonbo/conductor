"""A tool that exposes RAG retrieval to the agent.

Calls Retriever.retrieve() directly and hands the agent raw chunks — never
RagPipeline.answer(), which does its own LLM generation and would mean a
nested, redundant LLM call inside a tool call. Mirrors RecallTool's shape:
read-only, no approval, formatted string result, explicit "nothing found"
fallback.
"""
from __future__ import annotations

from typing import Any

from harness.core.rag.ports import Retriever, VectorStore
from harness.core.types import ToolSpec

_DESCRIPTION = (
    "Search the ingested document knowledge base (papers, files) for "
    "information relevant to a question. Use this for questions about "
    "document content — distinct from 'recall', which is personal facts "
    "about the user. Never guess at document content without calling "
    "this first."
)

_MAX_K = 20


class SearchDocumentsTool:
    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        default_collection: str,
        default_k: int = 5,
        collections: list[str] | None = None,
    ) -> None:
        self._retriever = retriever
        self._vector_store = vector_store
        self._default_collection = default_collection
        self._default_k = default_k
        self._collections = collections or []

    @property
    def spec(self) -> ToolSpec:
        properties: dict[str, Any] = {
            "query": {"type": "string", "description": "What to search for"},
            "k": {
                "type": "integer",
                "description": "Max chunks to retrieve",
                "default": self._default_k,
                "minimum": 1,
                "maximum": _MAX_K,
            },
        }
        if len(self._collections) > 1:
            properties["collection"] = {
                "type": "string",
                "enum": self._collections,
                "description": "Which document collection to search",
                "default": self._default_collection,
            }
        return ToolSpec(
            name="search_documents",
            description=_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        query = arguments["query"]
        k = max(1, min(int(arguments.get("k", self._default_k)), _MAX_K))
        collection = arguments.get("collection", self._default_collection)

        hits = await self._retriever.retrieve(query, k=k, collection=collection)
        if not hits:
            count = await self._vector_store.count(collection=collection)
            if count == 0:
                return f"No documents found in collection '{collection}'."
            return "No relevant documents found."

        lines = []
        for i, sc in enumerate(hits, start=1):
            section = " > ".join(sc.chunk.section_path) or "n/a"
            lines.append(
                f"[{i}] (score: {sc.score:.2f}, source: {sc.chunk.source_path}, "
                f"section: {section})\n{sc.chunk.text}"
            )
        return "\n\n".join(lines)
