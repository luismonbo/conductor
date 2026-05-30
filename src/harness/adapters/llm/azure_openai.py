"""Azure OpenAI adapter implementing core.llm.client.LLMClient.

Uses the Chat Completions API (supported indefinitely, and shape-compatible
with Ollama/vLLM — which is exactly why it's the right dev backend for a
harness you intend to retarget to Gemma-4). Native tool-calling is normalized
into core ToolCall/LLMResponse types so the agent never sees an SDK object.

The ToolCallParser is injected: pass NativeToolCallParser here, and a
PromptedToolCallParser into the Ollama adapter for small local models.
"""
from __future__ import annotations

from harness.adapters.llm.openai_wire import (
    build_request_messages,
    parse_completion,
    spec_to_wire,
)
from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.types import LLMResponse, Message, ToolSpec


class AzureOpenAIClient(LLMClient):
    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_version: str,
        parser: ToolCallParser,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        # Imported lazily so core/tests don't require the SDK installed.
        from openai import AsyncAzureOpenAI

        if api_key:
            self._client = AsyncAzureOpenAI(
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
            self._client = AsyncAzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
            )
        self._deployment = deployment
        self._parser = parser
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._deployment

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        wire_messages = build_request_messages(messages, self._parser, tools)

        kwargs: dict = {
            "model": self._deployment,
            "messages": wire_messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = [spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"

        completion = await self._client.chat.completions.create(**kwargs)
        return parse_completion(completion)
