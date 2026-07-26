"""Routes a file to docling or markitdown by format, falling back to
markitdown if docling fails — a single bad PDF must not abort an ingestion
run. See "Parser routing" in
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

from pathlib import Path

from harness.core.rag.document import ParsedContent
from harness.core.rag.ports import Parser

_DOCLING_EXTENSIONS = {".pdf"}


class ParserRouter:
    def __init__(self, docling: Parser, markitdown: Parser) -> None:
        self._docling = docling
        self._markitdown = markitdown

    async def parse(self, path: Path) -> ParsedContent:
        if path.suffix.lower() in _DOCLING_EXTENSIONS:
            try:
                return await self._docling.parse(path)
            except Exception:
                return await self._markitdown.parse(path)
        return await self._markitdown.parse(path)
