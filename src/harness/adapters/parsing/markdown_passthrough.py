"""Passthrough Parser for native markdown files: no conversion needed, the
file text already is the pipeline's working format. See "Parser routing" in
docs/superpowers/specs/2026-07-25-rag-ingestion-retrieval-design.md."""
from __future__ import annotations

from pathlib import Path

from harness.core.rag.document import ParsedContent


class MarkdownPassthroughParser:
    async def parse(self, path: Path) -> ParsedContent:
        return ParsedContent(text=path.read_text(encoding="utf-8"),
                             format=path.suffix.lstrip(".").lower() or "md", parser="markdown")
