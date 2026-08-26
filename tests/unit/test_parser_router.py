from __future__ import annotations

from pathlib import Path

import pytest

from harness.adapters.parsing.router import ParserRouter
from harness.core.rag.document import ParsedContent


class _StubParser:
    def __init__(self, result: ParsedContent):
        self._result = result
        self.calls: list[Path] = []

    async def parse(self, path: Path) -> ParsedContent:
        self.calls.append(path)
        return self._result


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["paper.pdf", "report.docx", "notes.txt"])
async def test_router_delegates_to_markitdown_regardless_of_format(filename):
    markitdown = _StubParser(ParsedContent(text="from markitdown", format="x", parser="markitdown"))
    router = ParserRouter(markitdown=markitdown)

    result = await router.parse(Path(filename))

    assert result.text == "from markitdown"
    assert markitdown.calls == [Path(filename)]
