from __future__ import annotations

from pathlib import Path

import pytest

from harness.adapters.parsing.router import ParserRouter
from harness.core.rag.document import ParsedContent


class _StubParser:
    def __init__(self, result: ParsedContent | None = None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list[Path] = []

    async def parse(self, path: Path) -> ParsedContent:
        self.calls.append(path)
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result


@pytest.mark.asyncio
async def test_pdf_routes_to_docling():
    docling = _StubParser(ParsedContent(text="from docling", format="pdf", parser="docling"))
    markitdown = _StubParser(ParsedContent(text="from markitdown", format="pdf", parser="markitdown"))
    router = ParserRouter(docling=docling, markitdown=markitdown)

    result = await router.parse(Path("paper.pdf"))

    assert result.text == "from docling"
    assert docling.calls == [Path("paper.pdf")]
    assert markitdown.calls == []


@pytest.mark.asyncio
async def test_non_pdf_routes_to_markitdown():
    docling = _StubParser(ParsedContent(text="x", format="docx", parser="docling"))
    markitdown = _StubParser(ParsedContent(text="from markitdown", format="docx", parser="markitdown"))
    router = ParserRouter(docling=docling, markitdown=markitdown)

    result = await router.parse(Path("report.docx"))

    assert result.text == "from markitdown"
    assert docling.calls == []


@pytest.mark.asyncio
async def test_docling_failure_falls_back_to_markitdown():
    docling = _StubParser(error=RuntimeError("docling blew up"))
    markitdown = _StubParser(ParsedContent(text="fallback text", format="pdf", parser="markitdown"))
    router = ParserRouter(docling=docling, markitdown=markitdown)

    result = await router.parse(Path("weird.pdf"))

    assert result.text == "fallback text"
    assert docling.calls == [Path("weird.pdf")]
    assert markitdown.calls == [Path("weird.pdf")]
