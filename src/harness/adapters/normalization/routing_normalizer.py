"""Dispatches normalization by which parser produced the ParsedContent: docling
output (structured JSON) goes to DoclingNormalizer, everything else (markdown
text, markitdown's markdown-ish rendering) goes to MarkdownNormalizer. See
"Parser routing" in docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""

from __future__ import annotations

from harness.core.rag.document import NormalizedDocument, ParsedContent
from harness.core.rag.ports import Normalizer


class RoutingNormalizer:
    def __init__(self, markdown: Normalizer, docling: Normalizer) -> None:
        self._markdown = markdown
        self._docling = docling

    async def normalize(
        self, parsed: ParsedContent, source_path: str, collection: str
    ) -> list[NormalizedDocument]:
        target = self._docling if parsed.parser == "docling" else self._markdown
        return await target.normalize(parsed, source_path, collection)
