from __future__ import annotations

import pytest

from harness.adapters.parsing.markitdown_parser import MarkitdownParseError, MarkitdownParser


@pytest.mark.asyncio
async def test_parses_html_file_to_markdown_text(tmp_path):
    html_path = tmp_path / "note.html"
    html_path.write_text(
        "<html><head><title>Note</title></head>"
        "<body><h1>Heading</h1><p>Some body text.</p></body></html>"
    )
    parser = MarkitdownParser()

    parsed = await parser.parse(html_path)

    assert "Heading" in parsed.text
    assert "Some body text" in parsed.text
    assert parsed.format == "html"
    assert parsed.parser == "markitdown"


@pytest.mark.asyncio
async def test_raises_markitdown_parse_error_on_missing_file(tmp_path):
    parser = MarkitdownParser()
    with pytest.raises(MarkitdownParseError):
        await parser.parse(tmp_path / "does_not_exist.html")
