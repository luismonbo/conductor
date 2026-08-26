"""Parses every file via markitdown. Previously routed a subset of formats to
docling too, but DOCLING_EXTENSIONS had been empty since a macOS OpenMP
crash (torch + onnxruntime both bundle libomp — see git history for
docling_parser.py) — never reachable, pure dead weight, so it was removed
entirely rather than kept disabled. Reintroducing it later just means
adding a second Parser back into parse() and picking which extensions route
to it. See "Parser routing" in
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

from pathlib import Path

from harness.core.rag.document import ParsedContent
from harness.core.rag.ports import Parser


class ParserRouter:
    def __init__(self, markitdown: Parser) -> None:
        self._markitdown = markitdown

    async def parse(self, path: Path) -> ParsedContent:
        return await self._markitdown.parse(path)
