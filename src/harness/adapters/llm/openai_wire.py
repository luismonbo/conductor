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
from harness.core.llm.errors import ContentFiltered
from harness.core.llm.tool_parsing import ToolCallParser
from harness.core.types import LLMResponse, Message, Role, ToolCall, ToolSpec

# The graph's LangChain CallbackHandler only sees LangGraph's own node-level
# spans — it has no visibility into a plain async function's raw SDK call.
# `langfuse.openai` closes that gap by patching AsyncOpenAI/AsyncAzureOpenAI
# in place so `chat.completions.create()` itself becomes a `generation`
# observation (model, tokens, cost), nested under the active trace. The
# patch is process-wide and permanent once imported, so callers below only
# use `langfuse_configured()` to decide whether to import it at all —
# keeping credential-free test runs from ever touching langfuse.


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


def _content_filter_categories(error_body: Any) -> tuple[str, ...]:
    """The categories a provider's content filter flagged, from the error object
    of its 400 (Azure: innererror.content_filter_result); () when absent."""
    inner = error_body.get("innererror") if isinstance(error_body, dict) else None
    results = inner.get("content_filter_result") if isinstance(inner, dict) else None
    if not isinstance(results, dict):
        return ()
    return tuple(
        name
        for name, verdict in results.items()
        if isinstance(verdict, dict) and verdict.get("filtered")
    )


async def _create_completion(client: Any, kwargs: dict[str, Any]) -> Any:
    """chat.completions.create, with a content-filter refusal raised as the
    domain's ContentFiltered. Duck-typed like the rest of this module: the SDK's
    API errors carry the provider's `code` and error `body`."""
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as exc:
        if getattr(exc, "code", None) == "content_filter":
            raise ContentFiltered(
                "prompt", _content_filter_categories(getattr(exc, "body", None))
            ) from exc
        raise


class _OpenAIBaseClient(LLMClient):
    """Shared generate() body for all OpenAI Chat Completions-shaped adapters.

    Subclasses build their SDK client (AsyncOpenAI / AsyncAzureOpenAI) and
    call super().__init__(client, model_id, parser, temperature). They inherit
    generate() and model_id for free with no duplication.
    """

    # Request parameter that caps completion tokens. llama-server, vLLM and
    # LiteLLM accept max_tokens; Azure's reasoning-model deployments require
    # max_completion_tokens (AzureOpenAIClient overrides this).
    _OUTPUT_LIMIT_PARAM = "max_tokens"

    def __init__(
        self,
        client: Any,
        model_id: str,
        parser: ToolCallParser,
        temperature: float,
        max_output_tokens: int = 0,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._parser = parser
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens  # 0 = no limit sent

    @property
    def model_id(self) -> str:
        return self._model_id

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        wire_messages = build_request_messages(messages, self._parser, tools)
        kwargs: dict = {
            "model": model or self._model_id,
            "messages": wire_messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = [spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"
        if self._max_output_tokens:
            kwargs[self._OUTPUT_LIMIT_PARAM] = self._max_output_tokens
        completion = await _create_completion(self._client, kwargs)
        return parse_completion(completion)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str | LLMResponse, None]:
        wire_messages = build_request_messages(messages, self._parser, tools)
        kwargs: dict = {
            "model": model or self._model_id,
            "messages": wire_messages,
            "temperature": self._temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [spec_to_wire(s) for s in tools]
            kwargs["tool_choice"] = "auto"

        kwargs["stream_options"] = {"include_usage": True}
        if self._max_output_tokens:
            kwargs[self._OUTPUT_LIMIT_PARAM] = self._max_output_tokens

        text_acc: list[str] = []
        tc_acc: dict[int, dict] = {}  # chunk index → accumulated data
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        stream_obj = await _create_completion(self._client, kwargs)
        async for chunk in stream_obj:
            if getattr(chunk, "usage", None) and chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
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
            usage=usage,
        )
