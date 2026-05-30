"""Azure OpenAI adapter implementing core.llm.client.LLMClient.

Uses the Chat Completions API (supported indefinitely, and shape-compatible
with Ollama/vLLM — which is exactly why it's the right dev backend for a
harness you intend to retarget to Gemma-4). Native tool-calling is normalized
into core ToolCall/LLMResponse types so the agent never sees an SDK object.

The ToolCallParser is injected: pass NativeToolCallParser here, and a
PromptedToolCallParser into the Ollama adapter for small local models.
"""
from __future__ import annotations

import json
import uuid

from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.types import (
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolSpec,
)


def _message_to_wire(msg: Message) -> dict:
    """Translate a core Message into the OpenAI chat wire format."""
    if msg.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }
    out: dict = {"role": msg.role.value, "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments),
                },
            }
            for c in msg.tool_calls
        ]
    return out


def _spec_to_wire(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


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
        wire_messages = [_message_to_wire(m) for m in messages]

        # The native parser returns "" here; a prompted parser would inject a
        # format spec. Harmless either way — keeps the adapter parser-agnostic.
        if tools:
            addendum = self._parser.system_prompt_addendum(tools)
            if addendum and wire_messages and wire_messages[0]["role"] == "system":
                wire_messages[0]["content"] += "\n\n" + addendum

        kwargs: dict = {
            "model": self._deployment,
            "messages": wire_messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = [_spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: tuple[ToolCall, ...] = ()
        if msg.tool_calls:
            parsed = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                parsed.append(
                    ToolCall(
                        id=tc.id or str(uuid.uuid4()),
                        name=tc.function.name,
                        arguments=args,
                    )
                )
            tool_calls = tuple(parsed)

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
        )
