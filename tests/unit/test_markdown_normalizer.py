import pytest

from harness.adapters.normalization.markdown_normalizer import MarkdownNormalizer
from harness.core.rag.document import ParsedContent


def _parsed(text: str) -> ParsedContent:
    return ParsedContent(text=text, format="md", parser="markdown")


@pytest.mark.asyncio
async def test_splits_on_atx_headings_with_levels():
    text = "# Title\n\nlead\n\n## A\n\nalpha\n\n### A.1\n\nnested\n"
    [doc] = await MarkdownNormalizer().normalize(_parsed(text), "docs/x.md", "docs")
    assert doc.title == "Title"
    assert [(s.title, s.level) for s in doc.sections] == [
        ("Title", 1), ("A", 2), ("A.1", 3),
    ]
    assert doc.sections[1].text.strip() == "alpha"
    assert doc.document_id.startswith("docs/")


@pytest.mark.asyncio
async def test_leading_content_before_first_heading_becomes_level0_section():
    text = "intro paragraph\n\n# Heading\n\nbody\n"
    [doc] = await MarkdownNormalizer().normalize(_parsed(text), "docs/y.md", "docs")
    assert doc.sections[0].level == 0
    assert doc.sections[0].text.strip() == "intro paragraph"


@pytest.mark.asyncio
async def test_no_headings_yields_single_level0_section():
    [doc] = await MarkdownNormalizer().normalize(_parsed("just text\n"), "docs/z.md", "docs")
    assert len(doc.sections) == 1
    assert doc.sections[0].level == 0
