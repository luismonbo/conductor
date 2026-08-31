import pytest

from harness.adapters.normalization.routing_normalizer import RoutingNormalizer
from harness.core.rag.document import ParsedContent


class _Spy:
    def __init__(self, tag):
        self.tag = tag
        self.called = False

    async def normalize(self, parsed, source_path, collection):
        self.called = True
        return [self.tag]


@pytest.mark.asyncio
async def test_routes_docling_parser_to_docling_normalizer():
    md, dl = _Spy("md"), _Spy("dl")
    r = RoutingNormalizer(markdown=md, docling=dl)
    out = await r.normalize(
        ParsedContent(text="{}", format="pdf", parser="docling"), "a.pdf", "papers"
    )
    assert out == ["dl"] and dl.called and not md.called


@pytest.mark.asyncio
async def test_routes_everything_else_to_markdown_normalizer():
    md, dl = _Spy("md"), _Spy("dl")
    r = RoutingNormalizer(markdown=md, docling=dl)
    out = await r.normalize(
        ParsedContent(text="# H", format="md", parser="markdown"), "a.md", "docs"
    )
    assert out == ["md"] and md.called and not dl.called
