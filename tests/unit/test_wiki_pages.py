"""Parsing wiki pages, and building the wiki from a source."""
from __future__ import annotations

import logging

import pytest

from harness.adapters.wiki.pages import build_wiki, parse_page

_WITH_FRONTMATTER = """---
type: concept
title: Generators
tags: [python]
---
# Generators

Functions that yield.
"""


def test_strips_the_frontmatter_and_takes_the_title_from_it():
    page = parse_page("concepts/generators", _WITH_FRONTMATTER)
    assert page.title == "Generators"
    assert page.body.startswith("# Generators")
    assert "type: concept" not in page.body


def test_a_page_without_frontmatter_is_titled_by_its_name():
    page = parse_page("about", "Luis builds agents.\n")
    assert (page.title, page.body) == ("about", "Luis builds agents.")


@pytest.mark.parametrize(
    ("frontmatter", "title"),
    [("title: About Luis", "About Luis"), ("title: 2024", "2024"), ("title:", "about"),
     ("title: '  '", "about"), ("type: overview", "about")],
)
def test_any_title_value_becomes_text_or_falls_back_to_the_name(frontmatter, title):
    page = parse_page("about", f"---\n{frontmatter}\n---\nBody.\n")
    assert (page.title, page.body) == (title, "Body.")


def test_a_page_with_windows_line_endings_still_loses_its_frontmatter():
    page = parse_page("about", "---\r\ntitle: About Luis\r\n---\r\nLuis builds agents.\r\n")
    assert (page.title, page.body) == ("About Luis", "Luis builds agents.")


@pytest.mark.parametrize(
    "text",
    [
        "---\ntitle: [unclosed\n---\nBody.\n",
        "---\ntitle: Never closed\nBody.\n",
        "---\ntitle: Broken\nupdated: 2026-02-30\n---\nBody.\n",  # PyYAML raises ValueError
    ],
)
def test_frontmatter_that_does_not_parse_keeps_the_whole_page(text, caplog):
    with caplog.at_level(logging.WARNING, logger="harness.adapters.wiki.pages"):
        page = parse_page("broken", text)
    assert (page.title, page.body) == ("broken", text.strip())
    assert "broken" in caplog.text


def test_a_block_that_does_not_open_the_page_is_plain_text():
    text = "# Research Log\n\n---\ntitle: Log\n---\n- entry\n"
    assert parse_page("log", text).body == text.strip()


class _StaticSource:
    label = "a test source"

    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def load(self) -> dict[str, str]:
        return self._pages


def test_builds_a_wiki_and_logs_how_many_pages_it_loaded(caplog):
    source = _StaticSource({"index": "- [[about]] — who Luis is", "about": _WITH_FRONTMATTER})
    with caplog.at_level(logging.INFO, logger="harness.adapters.wiki.pages"):
        wiki = build_wiki(source)
    assert wiki.page("about").title == "Generators"
    assert "wiki loaded: 2 pages from a test source" in caplog.text
    assert "Functions that yield" not in caplog.text  # never page content


def test_a_source_without_an_index_fails_the_build():
    with pytest.raises(ValueError, match="index"):
        build_wiki(_StaticSource({"about": "Luis."}))


def test_an_index_longer_than_a_read_warns_that_pages_are_unreachable(caplog):
    long_index = "- [[about]] — who Luis is\n" * 400  # well past MAX_READ_CHARS
    with caplog.at_level(logging.WARNING, logger="harness.adapters.wiki.pages"):
        build_wiki(_StaticSource({"index": long_index, "about": "Luis."}))
    assert f"wiki index is {len(long_index.strip())} characters" in caplog.text
