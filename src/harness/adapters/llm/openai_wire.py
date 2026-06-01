"""OpenAI Chat Completions wire-format helpers and shared adapter base.

`message_to_wire`, `spec_to_wire`, `build_request_messages`, `parse_completion`
are pure functions shared by every adapter that speaks the OpenAI shape (Azure,
llama-server, vLLM, Ollama). They import no provider SDK; `parse_completion`
duck-types the response so helpers are testable with plain SimpleNamespace stubs.

`_OpenAIBaseClient` owns the single `generate()` body that all OpenAI-shaped
adapters reuse. Subclasses only need to build the right SDK client and call
`super().__init__(client, model_id, parser, temperature)`.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator

from harness.core.llm.client import LLMClient
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.types import LLMResponse, Message, Role, ToolCall, ToolSpec


def message_to_wire(msg: Message) -> dict[str, Any]:
    """Translate a core Message into the OpenAI chat wire format."""
    if msg.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }
    out: dict[str, Any] = {"role": msg.role.value, "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in msg.tool_calls
        ]
    return out


def spec_to_wire(spec: ToolSpec) -> dict[str, Any]:
    """Translate a ToolSpec into the OpenAI chat function schema format."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def build_request_messages(
    messages: list[Message],
    parser: ToolCallParser,
    tools: list[ToolSpec] | None,
) -> list[dict[str, Any]]:
    """Translate messages and fold the parser's addendum into the system turn.

    Returns a fresh list of wire dicts; the input Messages are never mutated.
    The native parser returns an empty addendum, so this is a no-op there.
    """
    wire = [message_to_wire(m) for m in messages]
    if tools:
        addendum = parser.system_prompt_addendum(tools)
        if addendum and wire and wire[0]["role"] == "system":
            wire[0] = {**wire[0], "content": wire[0]["content"] + "\n\n" + addendum}
    return wire


def _decode_args(tc: Any) -> dict:
    try:
        return json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def _decode_args_str(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def parse_completion(completion: Any) -> LLMResponse:
    """Turn an OpenAI ChatCompletion-shaped object into a core LLMResponse."""
    choice = completion.choices[0]
    msg = choice.message

    tool_calls: tuple[ToolCall, ...] = ()
    if getattr(msg, "tool_calls", None):
        tool_calls = tuple(
            ToolCall(id=tc.id or str(uuid.uuid4()), name=tc.function.name, arguments=_decode_args(tc))
            for tc in msg.tool_calls
        )

    usage: dict[str, int] = {}
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


class _OpenAIBaseClient(LLMClient):
    """Shared generate() body for all OpenAI Chat Completions-shaped adapters.

    Subclasses build their SDK client (AsyncOpenAI / AsyncAzureOpenAI) and
    call super().__init__(client, model_id, parser, temperature). They inherit
    generate() and model_id for free with no duplication.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        parser: ToolCallParser,
        temperature: float,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._parser = parser
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model_id

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        wire_messages = build_request_messages(messages, self._parser, tools)
        kwargs: dict = {
            "model": self._model_id,
            "messages": wire_messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = [spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"
        completion = await self._client.chat.completions.create(**kwargs)
        return parse_completion(completion)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[str | LLMResponse, None]:
        wire_messages = build_request_messages(messages, self._parser, tools)
        kwargs: dict = {
            "model": self._model_id,
            "messages": wire_messages,
            "temperature": self._temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"

        text_acc: list[str] = []
        tc_acc: dict[int, dict] = {}  # chunk index → accumulated data
        finish_reason: str | None = None

        stream_obj = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream_obj:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                text_acc.append(delta.content)
                yield delta.content

            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    if not tc_delta.function:
                        continue
                    idx = tc_delta.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "args_parts": []}
                    if tc_delta.id:
                        tc_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function.name:
                        tc_acc[idx]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc_acc[idx]["args_parts"].append(tc_delta.function.arguments)

        tool_calls: tuple[ToolCall, ...] = ()
        if tc_acc:
            tool_calls = tuple(
                ToolCall(
                    id=data["id"] or str(uuid.uuid4()),
                    name=data["name"],
                    arguments=_decode_args_str("".join(data["args_parts"])),
                )
                for _, data in sorted(tc_acc.items())
            )

        yield LLMResponse(
            text="".join(text_acc),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
