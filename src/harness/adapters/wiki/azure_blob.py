"""A wiki read from a private Azure Blob container: production.

AzureBlobStorageLoader (langchain-azure-storage, Beta) signs in with
DefaultAzureCredential: a Container App with its managed identity, holding
Storage Blob Data Reader on the container; a developer with `az login`. The
package is imported only in load(), so nothing else pays for it. See
docs/superpowers/specs/2026-10-13-wiki-search-design.md.
"""
from __future__ import annotations

from urllib.parse import unquote, urlparse


class AzureBlobWikiSource:
    """The container's `.md` blobs under `prefix`, named without it or `.md`."""

    def __init__(self, account_url: str, container: str, prefix: str = "") -> None:
        self._account_url = account_url
        self._container = container
        # A folder: "wiki" and "/wiki/" both list "wiki/", never a sibling such
        # as "wiki-drafts/" (the listing matches names that start with the prefix).
        folder = prefix.strip("/")
        self._prefix = f"{folder}/" if folder else ""
        self.label = f"Azure Blob container {container!r}"

    def load(self) -> dict[str, str]:
        from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

        loader = AzureBlobStorageLoader(
            self._account_url, self._container, prefix=self._prefix or None
        )
        pages: dict[str, str] = {}
        for document in loader.lazy_load():
            blob_name = _blob_name(document.metadata["source"])
            if not blob_name.endswith(".md"):
                continue  # the wiki is markdown only; this is just a guard
            name = blob_name.removeprefix(self._prefix).lstrip("/").removesuffix(".md")
            pages[name] = document.page_content
        return pages


def _blob_name(source_url: str) -> str:
    """The blob's name: the URL path after the container, decoded."""
    return unquote(urlparse(source_url).path).split("/", 2)[2]
