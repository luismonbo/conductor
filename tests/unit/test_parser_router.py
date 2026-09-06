from __future__ import annotations

from pathlib import Path

import pytest

from harness.adapters.parsing.markdown_passthrough import MarkdownPassthroughParser
from harness.core.rag.document import ParsedContent


@pytest.mark.asyncio
async def test_markdown_passthrough_reads_file_text(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Hello\n\nworld")
    parsed = await MarkdownPassthroughParser().parse(p)
    assert parsed.parser == "markdown" and "# Hello" in parsed.text


@pytest.mark.asyncio
async def test_router_dispatches_by_suffix(tmp_path):
    class _P:
        def __init__(self, tag):
            self.tag = tag

        async def parse(self, path):
            return ParsedContent(text=self.tag, format="x", parser=self.tag)

    from harness.adapters.parsing.router import ParserRouter

    router = ParserRouter(
        markitdown=_P("markitdown"), markdown=_P("markdown"), docling=_P("docling")
    )
    assert (await router.parse(Path("a.pdf"))).parser == "docling"
    assert (await router.parse(Path("a.md"))).parser == "markdown"
    assert (await router.parse(Path("a.docx"))).parser == "markitdown"
