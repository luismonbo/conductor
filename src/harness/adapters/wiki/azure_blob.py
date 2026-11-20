"""A wiki read from a private Azure Blob container: production.

azure-storage-blob's ContainerClient signs in with DefaultAzureCredential: a
Container App with its managed identity, holding Storage Blob Data Reader on the
container; a developer with `az login`. The SDK is imported only in load(), so
nothing else pays for it. See docs/superpowers/specs/2026-10-13-wiki-search-design.md.
"""
from __future__ import annotations


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
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient

        pages: dict[str, str] = {}
        with (
            DefaultAzureCredential() as credential,
            ContainerClient(
                self._account_url, self._container, credential=credential
            ) as client,
        ):
            for blob in client.list_blobs(name_starts_with=self._prefix or None):
                if not blob.name.endswith(".md"):
                    continue  # the wiki is markdown only; this is just a guard
                name = blob.name.removeprefix(self._prefix).lstrip("/").removesuffix(".md")
                pages[name] = client.download_blob(blob.name, encoding="UTF-8").readall()
        return pages
