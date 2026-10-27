"""build_tool_deps loads the wiki next to RAG; build_wiki_source picks its source."""
from __future__ import annotations

import logging

from harness.adapters.wiki.azure_blob import AzureBlobWikiSource
from harness.adapters.wiki.local import LocalWikiSource
from harness.config.settings import Settings
from harness.orchestration.build import build_tool_deps, build_wiki_source


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        api_key="test-key",
        input_guard_enabled=False,
        llm_backend="fake",
        embedding_backend="fake",
        memory_backend="in_memory",
        rag_vector_store_backend="in_memory",
    )
    return Settings(**{**base, **overrides})


def _wiki_dir(tmp_path):
    (tmp_path / "index.md").write_text("- [[about]] — who Luis is\n")
    (tmp_path / "about.md").write_text("---\ntitle: About Luis\n---\nLuis builds agents.\n")
    return tmp_path


def test_picks_the_configured_source():
    assert build_wiki_source(_settings()) is None
    assert isinstance(build_wiki_source(_settings(wiki_dir="/srv/wiki")), LocalWikiSource)
    blob = _settings(
        wiki_blob_account_url="https://wikistore.blob.core.windows.net",
        wiki_blob_container="wiki-pages",
    )
    assert isinstance(build_wiki_source(blob), AzureBlobWikiSource)


def test_no_source_means_no_wiki_and_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="harness.orchestration.build"):
        deps = build_tool_deps(_settings())
    assert (deps.wiki, deps.wiki_error) == (None, None)
    assert "Wiki" not in caplog.text


def test_loads_the_wiki_from_a_local_folder(tmp_path):
    deps = build_tool_deps(_settings(wiki_dir=str(_wiki_dir(tmp_path))))
    assert deps.wiki.page("about").title == "About Luis"
    assert deps.wiki_error is None


def test_a_wiki_that_fails_to_load_is_recorded_with_a_warning(tmp_path, caplog):
    (tmp_path / "about.md").write_text("no index here")
    with caplog.at_level(logging.WARNING, logger="harness.orchestration.build"):
        deps = build_tool_deps(_settings(wiki_dir=str(tmp_path)))
    assert deps.wiki is None
    assert isinstance(deps.wiki_error, ValueError)
    assert "Wiki unavailable" in caplog.text


def test_the_wiki_loads_even_when_rag_is_unavailable(tmp_path):
    deps = build_tool_deps(
        _settings(wiki_dir=str(_wiki_dir(tmp_path)), rag_vector_store_backend="not_a_real_backend")
    )
    assert deps.retriever is None and deps.rag_error is not None
    assert deps.wiki is not None and deps.wiki_error is None
