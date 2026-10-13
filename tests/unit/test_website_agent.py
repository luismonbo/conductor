"""The website agent: the wiki tools only, and no start without its wiki."""
from __future__ import annotations

import pytest

from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.agents.definition import ToolDeps
from harness.agents.website.agent import AGENT
from harness.config.settings import Settings
from harness.core.wiki import Wiki, WikiPage


def _deps(wiki=None, wiki_error=None) -> ToolDeps:
    return ToolDeps(
        settings=Settings(_env_file=None, api_key="test-key", input_guard_enabled=False),
        long_term=InMemoryLongTerm(),
        retriever=None,
        vector_store=None,
        collections=(),
        wiki=wiki,
        wiki_error=wiki_error,
    )


def test_offers_exactly_the_two_wiki_tools_and_needs_no_rag():
    wiki = Wiki({"index": WikiPage("index", "Wiki Index", "- [[about]]")})
    assert AGENT.tools(_deps(wiki=wiki)).names() == ["read_wiki_index", "read_wiki_page"]


def test_does_not_start_without_its_wiki_and_shows_why():
    cause = FileNotFoundError("wiki folder /srv/wiki does not exist")
    with pytest.raises(RuntimeError, match="needs its wiki") as excinfo:
        AGENT.tools(_deps(wiki_error=cause))
    assert excinfo.value.__cause__ is cause


def test_the_prompt_sends_the_model_to_the_wiki():
    assert "wiki index" in AGENT.prompt
    assert "documents you can search" not in AGENT.prompt
