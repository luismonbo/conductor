"""A wiki read from a private Azure Blob container (fake SDK clients; no network)."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import azure.identity
import azure.storage.blob
import pytest
from azure.storage.blob import BlobProperties

from harness.adapters.wiki.azure_blob import AzureBlobWikiSource

_ACCOUNT = "https://wikistore.blob.core.windows.net"


class _FakeCredential:
    """Stands in for DefaultAzureCredential: records that it was closed."""

    def __init__(self) -> None:
        self.exited = False

    def __enter__(self) -> _FakeCredential:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.exited = True


class _FakeDownload:
    """Like StorageStreamDownloader: text with an encoding, bytes without one."""

    def __init__(self, data: bytes, encoding: str | None) -> None:
        self._data = data
        self._encoding = encoding

    def readall(self) -> str | bytes:
        return self._data.decode(self._encoding) if self._encoding else self._data


class _FakeContainerClient:
    """Stands in for ContainerClient over `blobs` (name -> text, stored as UTF-8):
    records how it was built, each listing's name_starts_with and each download,
    and, like the service, lists only the names that start with the prefix."""

    def __init__(
        self, blobs: dict[str, str], account_url: str, container_name: str,
        credential: object = None,
    ) -> None:
        self.built_with = (account_url, container_name)
        self.credential = credential
        self.listed_with: list[str | None] = []
        self.downloaded: list[str] = []
        self.exited = False
        self._blobs = blobs

    def __enter__(self) -> _FakeContainerClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.exited = True

    def list_blobs(self, name_starts_with: str | None = None) -> Iterator[BlobProperties]:
        self.listed_with.append(name_starts_with)
        names = [name for name in self._blobs if name.startswith(name_starts_with or "")]
        return iter([BlobProperties(name=name) for name in names])

    def download_blob(self, blob: str, *, encoding: str | None = None) -> _FakeDownload:
        self.downloaded.append(blob)
        return _FakeDownload(self._blobs[blob].encode("utf-8"), encoding)


class _FakeAzure:
    """The two SDK classes load() uses, faked over one container's `blobs`."""

    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}
        self.credentials: list[_FakeCredential] = []
        self.clients: list[_FakeContainerClient] = []

    def credential(self) -> _FakeCredential:
        self.credentials.append(_FakeCredential())
        return self.credentials[-1]

    def container_client(
        self, account_url: str, container_name: str, credential: object = None
    ) -> _FakeContainerClient:
        client = _FakeContainerClient(self.blobs, account_url, container_name, credential)
        self.clients.append(client)
        return client


@pytest.fixture
def azure_sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeAzure:
    fake = _FakeAzure()
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", fake.credential)
    monkeypatch.setattr(azure.storage.blob, "ContainerClient", fake.container_client)
    return fake


def test_names_pages_by_blob_name_without_md(azure_sdk):
    azure_sdk.blobs.update({"index.md": "the index", "concepts/generators.md": "generators"})
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load() == {
        "index": "the index",
        "concepts/generators": "generators",
    }


def test_strips_the_prefix_and_keeps_the_blob_name_as_is(azure_sdk):
    azure_sdk.blobs.update(
        {"wiki/index.md": "the index", "wiki/projects/agent harness.md": "harness"}
    )
    pages = AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki/").load()
    assert pages == {"index": "the index", "projects/agent harness": "harness"}


def test_a_prefix_without_its_slash_still_names_pages_from_the_wiki_root(azure_sdk):
    azure_sdk.blobs.update({"wiki/index.md": "the index"})
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki").load() == {
        "index": "the index"
    }


def test_a_prefix_without_its_slash_loads_only_that_folder(azure_sdk):
    azure_sdk.blobs.update({
        "wiki/index.md": "the index",
        "wiki-drafts/unpublished.md": "not public yet",
        "wiki.md": "beside the folder, not in it",
    })
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki").load() == {
        "index": "the index"
    }


def test_skips_blobs_that_are_not_markdown(azure_sdk):
    azure_sdk.blobs.update({"index.md": "the index", "notes.txt": "not a page"})
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load() == {"index": "the index"}
    assert azure_sdk.clients[0].downloaded == ["index.md"]


def test_lists_the_container_with_the_prefix_and_the_default_credential(azure_sdk):
    AzureBlobWikiSource(_ACCOUNT, "wiki-pages", prefix="wiki/").load()
    AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load()
    with_prefix, without = azure_sdk.clients
    assert with_prefix.built_with == without.built_with == (_ACCOUNT, "wiki-pages")
    assert with_prefix.credential is azure_sdk.credentials[0]
    assert without.credential is azure_sdk.credentials[1]
    assert (with_prefix.listed_with, without.listed_with) == (["wiki/"], [None])


def test_downloads_pages_as_utf8_text(azure_sdk):
    azure_sdk.blobs.update({"about.md": "Luis vive a Bologna — città"})
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load() == {
        "about": "Luis vive a Bologna — città"
    }


def test_closes_the_client_and_the_credential(azure_sdk):
    azure_sdk.blobs.update({"index.md": "the index"})
    AzureBlobWikiSource(_ACCOUNT, "wiki-pages").load()
    assert azure_sdk.clients[0].exited
    assert azure_sdk.credentials[0].exited


def test_the_label_names_the_container():
    assert AzureBlobWikiSource(_ACCOUNT, "wiki-pages").label == "Azure Blob container 'wiki-pages'"


def test_the_sdk_is_not_imported_until_a_load():
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys, harness.adapters.wiki.azure_blob; "
        "print('azure.storage.blob' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.stdout.strip() == "False"
