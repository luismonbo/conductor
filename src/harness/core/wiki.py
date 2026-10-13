"""The wiki an agent reads: pages held in memory, looked up by name.

A page's name is its path inside the wiki without `.md` (`concepts/generators`):
the name the wiki's own [[...]] links use. A name the model sends is only ever a
key in this mapping, never a file path, so no name can reach the filesystem.
See docs/superpowers/specs/2026-10-13-wiki-search-design.md.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

INDEX_PAGE = "index"

# The most one tool read returns, about 2k tokens: a page, or the index.
MAX_READ_CHARS = 8_000

_LINK = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass(frozen=True)
class WikiPage:
    name: str
    title: str
    body: str  # markdown, without the frontmatter


def normalize_page_name(raw: str) -> str:
    """The name of the page `raw` refers to, however the model wrote it.

    `[[concepts/generators|Generators]]`, a whole index line such as
    `- [[concepts/generators]] — what it covers`, `concepts/generators#Usage`
    and `concepts/generators.md` all name `concepts/generators`.
    """
    link = _LINK.search(raw)
    name = link.group(1) if link else raw
    name = name.split("|", 1)[0].split("#", 1)[0].strip()
    return name.removesuffix(".md")


class Wiki:
    """An immutable set of pages that includes the index."""

    def __init__(self, pages: Mapping[str, WikiPage]) -> None:
        if INDEX_PAGE not in pages:
            raise ValueError(f"the wiki has no {INDEX_PAGE}.md page")
        self._pages = MappingProxyType(dict(pages))

    def index(self) -> WikiPage:
        return self._pages[INDEX_PAGE]

    def page(self, name: str) -> WikiPage | None:
        return self._pages.get(normalize_page_name(name))

    def __len__(self) -> int:
        return len(self._pages)


class WikiSource(Protocol):
    """Where a wiki's pages come from: raw markdown keyed by page name."""

    label: str  # for the startup log, e.g. "a local folder"

    def load(self) -> dict[str, str]: ...
