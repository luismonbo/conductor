"""OpenAI-compatible LLM adapter implementing core.llm.client.LLMClient.

One adapter for every server that speaks the OpenAI Chat Completions API —
llama.cpp `llama-server` (local default), vLLM, Ollama, or any compatible
endpoint. The engine is selected purely by `base_url`, which is the whole
reason a dev-local Gemma on Metal and a future NVIDIA vLLM deployment need no
code change between them.

Wire-format translation is shared with the Azure adapter via openai_wire.py.
The `client` parameter exists for tests (inject a stand-in); in normal use it
is left None and a real AsyncOpenAI is built lazily so the `openai` SDK is only
required when this backend is actually selected.
"""
from __future__ import annotations

from typing import Any

from harness.adapters.llm.openai_wire import (
    build_request_messages,
    parse_completion,
    spec_to_wire,
)
from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.types import LLMResponse, Message, ToolSpec

# Most local servers (llama-server) ignore auth, but the SDK rejects an empty
# api_key, so we substitute a harmless placeholder when none is configured.
_PLACEHOLDER_KEY = "sk-no-key-required"


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        parser: ToolCallParser,
        api_key: str = "",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            # Imported lazily so core/tests don't require the SDK installed.
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url, api_key=api_key or _PLACEHOLDER_KEY)
        self._client = client
        self._model = model
        self._parser = parser
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        wire_messages = build_request_messages(messages, self._parser, tools)

        kwargs: dict = {
            "model": self._model,
            "messages": wire_messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = [spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"

        completion = await self._client.chat.completions.create(**kwargs)
        return parse_completion(completion)
