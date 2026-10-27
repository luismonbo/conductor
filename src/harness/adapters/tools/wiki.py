"""Read-only tools over the wiki: its index, and one page by name.

The page name the model sends is only a key in the in-memory Wiki, never a
path. Results carry page titles and text, which are public, and no path or
URL. See docs/superpowers/specs/2026-10-13-wiki-search-design.md.
"""
from __future__ import annotations

from typing import Any

from harness.core.tools.base import ToolError
from harness.core.types import ToolSpec
from harness.core.wiki import MAX_READ_CHARS, Wiki, WikiPage

_INDEX_DESCRIPTION = (
    "List the pages of the wiki about the site owner, one line each with what the "
    "page covers. Call this first, then read_wiki_page for the pages that answer "
    "the question."
)
_PAGE_DESCRIPTION = (
    "Read one page of the wiki about the site owner, by the name the wiki index "
    "gives it (for example: projects/agent-harness)."
)


def _capped(text: str) -> str:
    if len(text) <= MAX_READ_CHARS:
        return text
    return f"{text[:MAX_READ_CHARS]}\n\n[Truncated at {MAX_READ_CHARS:,} characters.]"


def _render(page: WikiPage) -> str:
    return _capped(f"# {page.title}\n\n{page.body}")


class ReadWikiIndexTool:
    def __init__(self, wiki: Wiki) -> None:
        self._wiki = wiki

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_wiki_index",
            description=_INDEX_DESCRIPTION,
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        return _capped(self._wiki.index().body)


class ReadWikiPageTool:
    def __init__(self, wiki: Wiki) -> None:
        self._wiki = wiki

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_wiki_page",
            description=_PAGE_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "page": {"type": "string", "description": "The page name, as the index gives it"},
                },
                "required": ["page"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        name = arguments.get("page")
        if not isinstance(name, str) or not name.strip():
            raise ToolError("page is required: give a page name from the wiki index")
        page = self._wiki.page(name)
        if page is None:
            raise ToolError(
                f"no page named {name.strip()!r}; the wiki index lists the pages "
                "(call read_wiki_index)"
            )
        return _render(page)
