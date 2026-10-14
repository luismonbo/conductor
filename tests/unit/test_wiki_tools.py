"""The wiki tools: read the index, read a page by name."""
from __future__ import annotations

import pytest

from harness.adapters.tools.wiki import ReadWikiIndexTool, ReadWikiPageTool
from harness.core.tools.base import ToolError
from harness.core.wiki import MAX_READ_CHARS, Wiki, WikiPage

_INDEX = WikiPage("index", "Wiki Index", "- [[projects/harness]] — the agent harness")
_HARNESS = WikiPage("projects/harness", "Agent Harness", "Luis built a production agent harness.")


def _wiki(*extra: WikiPage) -> Wiki:
    return Wiki({p.name: p for p in (_INDEX, _HARNESS, *extra)})


def test_the_specs_the_model_sees():
    index_spec = ReadWikiIndexTool(_wiki()).spec
    page_spec = ReadWikiPageTool(_wiki()).spec
    assert (index_spec.name, page_spec.name) == ("read_wiki_index", "read_wiki_page")
    assert index_spec.parameters == {"type": "object", "properties": {}}
    assert page_spec.parameters["required"] == ["page"]
    assert "read_wiki_page" in index_spec.description  # tells the model what comes next


async def test_reads_the_index():
    assert await ReadWikiIndexTool(_wiki()).run({}) == _INDEX.body


@pytest.mark.parametrize(
    "name",
    ["projects/harness", "[[projects/harness|the harness]]", "projects/harness.md",
     "- [[projects/harness]] — the agent harness"],
)
async def test_reads_a_page_by_any_spelling_of_its_name(name):
    result = await ReadWikiPageTool(_wiki()).run({"page": name})
    assert result == "# Agent Harness\n\nLuis built a production agent harness."


async def test_an_unknown_page_tells_the_model_to_check_the_index():
    with pytest.raises(ToolError, match=r"no page named 'projects/missing'.*read_wiki_index"):
        await ReadWikiPageTool(_wiki()).run({"page": "projects/missing"})


async def test_a_path_like_name_is_just_an_unknown_page():
    with pytest.raises(ToolError, match="no page named"):
        await ReadWikiPageTool(_wiki()).run({"page": "../../etc/passwd"})


@pytest.mark.parametrize("arguments", [{}, {"page": None}, {"page": "  "}, {"page": 7}])
async def test_a_missing_page_name_is_a_tool_error(arguments):
    with pytest.raises(ToolError, match="page is required"):
        await ReadWikiPageTool(_wiki()).run(arguments)


async def test_a_long_page_is_cut_with_a_note():
    long = WikiPage("long", "Long", "x" * (MAX_READ_CHARS * 2))
    result = await ReadWikiPageTool(_wiki(long)).run({"page": "long"})
    assert result.startswith("# Long\n\n")
    assert result.endswith("\n\n[Truncated at 8,000 characters.]")
    assert len(result) < MAX_READ_CHARS + 100


async def test_a_long_index_is_cut_too():
    index = WikiPage("index", "Wiki Index", "y" * (MAX_READ_CHARS + 1))
    result = await ReadWikiIndexTool(Wiki({"index": index})).run({})
    assert result.endswith("[Truncated at 8,000 characters.]")
