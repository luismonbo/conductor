"""Routes a file to docling or markitdown by format, falling back to
markitdown if docling fails — a single bad PDF must not abort an ingestion
run. See "Parser routing" in
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md.

DOCLING_EXTENSIONS is empty for now: torch (pulled in just by importing
docling) crashes on macOS the moment anything else in the process also links
its own OpenMP runtime — confirmed happening with both onnxruntime (docling's
own OCR) and, separately, pdfminer/cryptography (markitdown's PDF path) — and
it's a native crash (SIGABRT/SIGSEGV), not a Python exception, so the
try/except fallback below can't catch it and recover. build_parser_router()
(orchestration/build.py) checks this constant and skips importing docling at
all when it's empty — merely importing torch is enough to trigger the
conflict, calling docling is not required. Re-add ".pdf" once the conflict is
actually root-caused (which two libraries' OpenMP copies collide, and why);
the fallback-on-exception path (for recoverable docling errors) stays in
place either way. See docs/devlog/010."""
from __future__ import annotations

from pathlib import Path

from harness.core.rag.document import ParsedContent
from harness.core.rag.ports import Parser

DOCLING_EXTENSIONS: set[str] = set()


class ParserRouter:
    def __init__(self, docling: Parser, markitdown: Parser) -> None:
        self._docling = docling
        self._markitdown = markitdown

    async def parse(self, path: Path) -> ParsedContent:
        if path.suffix.lower() in DOCLING_EXTENSIONS:
            try:
                return await self._docling.parse(path)
            except Exception:
                return await self._markitdown.parse(path)
        return await self._markitdown.parse(path)
