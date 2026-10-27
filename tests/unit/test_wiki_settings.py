"""The wiki settings: one source per deployment."""
from __future__ import annotations

import pytest

from harness.config.settings import Settings
from harness.config.targets import resolve_target_overrides

_BASE = dict(_env_file=None, api_key="test-key", input_guard_enabled=False)
_BLOB = dict(
    wiki_blob_account_url="https://wikistore.blob.core.windows.net",
    wiki_blob_container="wiki-pages",
)


def test_no_wiki_by_default():
    s = Settings(**_BASE)
    assert (s.wiki_dir, s.wiki_blob_account_url, s.wiki_blob_container, s.wiki_blob_prefix) == (
        "", "", "", "",
    )


def test_a_local_folder_or_a_blob_container_is_fine():
    assert Settings(**_BASE, wiki_dir="/srv/wiki").wiki_dir == "/srv/wiki"
    assert Settings(**_BASE, **_BLOB, wiki_blob_prefix="wiki/").wiki_blob_prefix == "wiki/"


def test_both_sources_at_once_are_refused():
    with pytest.raises(ValueError, match="one wiki source"):
        Settings(**_BASE, **_BLOB, wiki_dir="/srv/wiki")


@pytest.mark.parametrize(
    "half",
    [{"wiki_blob_account_url": "https://wikistore.blob.core.windows.net"},
     {"wiki_blob_container": "wiki-pages"}],
)
def test_half_a_blob_source_is_refused(half):
    with pytest.raises(ValueError, match="HARNESS_WIKI_BLOB_ACCOUNT_URL and HARNESS_WIKI_BLOB_CONTAINER"):
        Settings(**_BASE, **half)


def test_a_prefix_without_a_container_is_refused():
    with pytest.raises(ValueError, match="HARNESS_WIKI_BLOB_PREFIX"):
        Settings(**_BASE, wiki_blob_prefix="wiki/")


def test_wiki_settings_are_allowed_in_a_target_file(tmp_path):
    (tmp_path / "site.yaml").write_text("wiki_blob_container: wiki-pages\n")
    assert resolve_target_overrides("site", base_dir=tmp_path) == {"wiki_blob_container": "wiki-pages"}
