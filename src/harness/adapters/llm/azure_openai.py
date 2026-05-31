"""Azure OpenAI adapter implementing core.llm.client.LLMClient.

Uses the Chat Completions API (supported indefinitely, and shape-compatible
with llama-server/vLLM — which is exactly why it's the right dev backend for
a harness you intend to retarget to Gemma-4). Native tool-calling is
normalized into core ToolCall/LLMResponse types so the agent never sees an SDK
object.

generate() and model_id are inherited from _OpenAIBaseClient (openai_wire.py).
The `client` parameter exists for tests (inject a stand-in, same seam as
OpenAICompatibleClient); in normal use it is left None and the SDK client is
built lazily from the deployment / endpoint / credentials configuration.
"""
from __future__ import annotations

from typing import Any

from harness.adapters.llm.openai_wire import _OpenAIBaseClient
from harness.core.llm.tool_parsing import ToolCallParser


class AzureOpenAIClient(_OpenAIBaseClient):
    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_version: str,
        parser: ToolCallParser,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
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
        super().__init__(client, deployment, parser, temperature)
