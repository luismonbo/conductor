"""Wiki pages from raw markdown: the YAML frontmatter comes off and its title
names the page. build_wiki loads a source into the in-memory Wiki."""
from __future__ import annotations

import logging

import yaml

from harness.core.wiki import MAX_READ_CHARS, Wiki, WikiPage, WikiSource

logger = logging.getLogger(__name__)

_FENCE = "---"


def parse_page(name: str, text: str) -> WikiPage:
    """A page without the frontmatter that opens it (a `---` block).

    The title is the frontmatter's `title`, else the page name. Frontmatter
    that doesn't parse stays in the page, with a warning: one malformed page
    must not take the wiki down.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FENCE:
        return WikiPage(name=name, title=name, body=text.strip())
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == _FENCE), None)
    meta = None
    if closing is not None:
        try:
            meta = yaml.safe_load("".join(lines[1:closing]))
        except Exception:  # YAMLError, or ValueError for a date such as 2026-02-30
            closing = None
    if closing is None:
        logger.warning("wiki page %r: its frontmatter doesn't parse; kept as text", name)
        return WikiPage(name=name, title=name, body=text.strip())
    raw_title = meta.get("title") if isinstance(meta, dict) else None
    title = str(raw_title).strip() if raw_title is not None else ""
    body = "".join(lines[closing + 1:]).strip()
    return WikiPage(name=name, title=title or name, body=body)


def build_wiki(source: WikiSource) -> Wiki:
    """Load every page from `source` into a Wiki; fails without an index page."""
    wiki = Wiki({name: parse_page(name, text) for name, text in source.load().items()})
    logger.info("wiki loaded: %d pages from %s", len(wiki), source.label)
    index_chars = len(wiki.index().body)
    if index_chars > MAX_READ_CHARS:
        logger.warning(
            "wiki index is %d characters but a read returns %d: pages listed after "
            "that are unreachable, so split the index",
            index_chars,
            MAX_READ_CHARS,
        )
    return wiki
