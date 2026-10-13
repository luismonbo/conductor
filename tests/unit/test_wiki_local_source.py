"""A wiki read from a folder on disk."""
from __future__ import annotations

import pytest

from harness.adapters.wiki.local import LocalWikiSource


def _write(root, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_names_pages_by_their_path_inside_the_wiki_without_md(tmp_path):
    _write(tmp_path, "index.md", "the index")
    _write(tmp_path, "concepts/generators.md", "generators")
    assert LocalWikiSource(tmp_path).load() == {
        "index": "the index",
        "concepts/generators": "generators",
    }


def test_reads_only_markdown_files(tmp_path):
    _write(tmp_path, "index.md", "the index")
    _write(tmp_path, "notes.txt", "not a page")
    _write(tmp_path, ".obsidian/workspace.json", "{}")
    assert set(LocalWikiSource(tmp_path).load()) == {"index"}


def test_skips_symlinks_so_nothing_outside_the_wiki_is_read(tmp_path):
    wiki, outside = tmp_path / "wiki", tmp_path / "outside"
    _write(wiki, "index.md", "the index")
    _write(outside, "secret.md", "private")
    (wiki / "linked.md").symlink_to(outside / "secret.md")
    (wiki / "linked_dir").symlink_to(outside, target_is_directory=True)
    assert set(LocalWikiSource(wiki).load()) == {"index"}


def test_a_page_that_is_not_utf8_fails_the_load_naming_it(tmp_path):
    _write(tmp_path, "index.md", "the index")
    (tmp_path / "latin.md").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ValueError, match="latin"):
        LocalWikiSource(tmp_path).load()


def test_a_missing_folder_fails_the_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalWikiSource(tmp_path / "nope").load()


def test_expands_a_home_relative_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write(tmp_path, "wiki/index.md", "the index")
    assert LocalWikiSource("~/wiki").load() == {"index": "the index"}
    assert LocalWikiSource("~/wiki").label == "a local folder"
