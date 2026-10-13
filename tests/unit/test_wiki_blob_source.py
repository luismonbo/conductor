"""A wiki read from a private Azure Blob container (a fake loader; no network)."""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from langchain_core.documents import Document

from harness.adapters.wiki.azure_blob import AzureBlobWikiSource

_ACCOUNT = "https://wikistore.blob.core.windows.net"


class _FakeLoader:
    """Stands in for AzureBlobStorageLoader: records its arguments, yields Documents."""

    created: list[tuple[tuple, dict]] = []
    blobs: dict[str, str] = {}

    def __init__(self, *args, **kwargs) -> None:
        _FakeLoader.created.append((args, kwargs))

    def lazy_load(self):
        for name, text in _FakeLoader.blobs.items():
            yield Document(page_content=text, metadata={"source": f"{_ACCOUNT}/wiki-pages/{name}"})


@pytest.fixture
def fake_loader(monkeypatch):
    _FakeLoader.created = []
    _FakeLoader.blobs = {}
    module = types.ModuleType("langchain_azure_storage.document_loaders")
    module.AzureBlobStorageLoader = _FakeLoader
    monkeypatch.setitem(sys.modules, "langchain_azure_storage.document_loaders", module)
    return _FakeLoader


def test_names_pages_by_blob_name_without_md(fake_loader):
    fake_loader.blobs = {"index.md": "the index", "concepts/generators.md": "generators"}
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load() == {
        "index": "the index",
        "concepts/generators": "generators",
    }


def test_strips_the_prefix_and_decodes_the_url(fake_loader):
    fake_loader.blobs = {"wiki/index.md": "the index", "wiki/projects/agent%20harness.md": "harness"}
    pages = AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki/").load()
    assert pages == {"index": "the index", "projects/agent harness": "harness"}


def test_a_prefix_without_its_slash_still_names_pages_from_the_wiki_root(fake_loader):
    fake_loader.blobs = {"wiki/index.md": "the index"}
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki").load() == {"index": "the index"}


def test_skips_blobs_that_are_not_markdown(fake_loader):
    fake_loader.blobs = {"index.md": "the index", "notes.txt": "not a page"}
    assert set(AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load()) == {"index"}


def test_asks_the_loader_for_the_container_and_prefix_with_the_default_credential(fake_loader):
    AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki/").load()
    AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load()
    assert fake_loader.created == [
        ((_ACCOUNT, "wiki-pages"), {"prefix": "wiki/"}),
        ((_ACCOUNT, "wiki-pages"), {"prefix": None}),
    ]


def test_the_label_names_the_container():
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages").label == "Azure Blob container 'wiki-pages'"


def test_the_sdk_is_not_imported_until_a_load():
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys, harness.adapters.wiki.azure_blob; "
        "print('langchain_azure_storage' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.stdout.strip() == "False"
