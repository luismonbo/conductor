"""The in-memory wiki: page names and lookups."""
from __future__ import annotations

import pytest

from harness.core.wiki import Wiki, WikiPage, normalize_page_name


def _page(name: str) -> WikiPage:
    return WikiPage(name=name, title=name.title(), body=f"body of {name}")


@pytest.mark.parametrize(
    "raw",
    [
        "concepts/generators",
        "  concepts/generators  ",
        "concepts/generators.md",
        "concepts/generators#Usage",
        "[[concepts/generators]]",
        "[[concepts/generators|Generators]]",
        "[[concepts/generators#Usage]]",
        "- [[concepts/generators]] — functions that yield",
    ],
)
def test_every_way_of_writing_a_page_name_means_the_same_page(raw):
    assert normalize_page_name(raw) == "concepts/generators"


def test_a_wiki_needs_an_index():
    with pytest.raises(ValueError, match="index"):
        Wiki({"about": _page("about")})


def test_looks_pages_up_by_any_spelling_of_their_name():
    wiki = Wiki({"index": _page("index"), "projects/harness": _page("projects/harness")})
    assert wiki.page("[[projects/harness|The harness]]") is wiki.page("projects/harness")
    assert wiki.page("projects/harness").body == "body of projects/harness"
    assert wiki.index().name == "index"
    assert len(wiki) == 2


def test_an_unknown_page_is_none():
    wiki = Wiki({"index": _page("index")})
    assert wiki.page("projects/missing") is None
    assert wiki.page("Index") is None  # names are case-sensitive


def test_a_path_like_name_is_only_an_unknown_key():
    assert Wiki({"index": _page("index")}).page("../../etc/passwd") is None


def test_the_wiki_does_not_change_when_its_source_mapping_does():
    pages = {"index": _page("index")}
    wiki = Wiki(pages)
    pages["late"] = _page("late")
    assert wiki.page("late") is None
