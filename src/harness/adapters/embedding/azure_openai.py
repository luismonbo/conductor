"""Embeddings via Azure OpenAI. Azure routes by *deployment* under
/openai/deployments/{deployment}/embeddings with api-version/auth semantics
the plain AsyncOpenAI client doesn't produce, so this mirrors
adapters/llm/azure_openai.py's client construction (api-key or managed
identity) rather than reusing OpenAICompatibleEmbedder.

The `client` parameter exists for tests (inject a stand-in, same seam as
AzureOpenAIClient); in normal use it is left None and the SDK client is
built lazily from the deployment / endpoint / credentials configuration.
"""
from __future__ import annotations

from typing import Any


class AzureOpenAIEmbedder:
    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_version: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = deployment
        if client is None:
            # Imported lazily so core/tests don't require the SDK installed.
            from openai import AsyncAzureOpenAI

            if api_key:
                client = AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                )
            else:
                # Managed identity path (preferred in Azure prod, per your stack).
                from azure.identity import (
                    DefaultAzureCredential,
                    get_bearer_token_provider,
                )

                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                client = AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                )
        self._client = client

    @property
    def model_id(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
