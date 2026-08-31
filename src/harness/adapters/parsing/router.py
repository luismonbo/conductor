"""Routes each file to a parser by extension: .pdf -> docling (heading- and
table-aware PDF structure; only actually importable in the Linux ingest
container, see docling_parser.py), .md/.markdown -> markdown passthrough,
everything else -> markitdown. See "Parser routing" in
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""

from __future__ import annotations

from pathlib import Path

from harness.core.rag.document import ParsedContent
from harness.core.rag.ports import Parser


class ParserRouter:
    def __init__(self, markitdown: Parser, markdown: Parser, docling: Parser) -> None:
        self._markitdown = markitdown
        self._markdown = markdown
        self._docling = docling

    async def parse(self, path: Path) -> ParsedContent:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return await self._docling.parse(path)
        if suffix in (".md", ".markdown"):
            return await self._markdown.parse(path)
        return await self._markitdown.parse(path)
