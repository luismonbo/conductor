"""Pure OpenAI chat wire-format translation.

Shared by every adapter that speaks the OpenAI Chat Completions shape
(Azure OpenAI, llama.cpp llama-server, vLLM, Ollama). Deliberately imports NO
provider SDK: `parse_completion` duck-types the response object, so these
helpers are unit-testable with a plain stand-in and impose no install cost on
the `fake` backend.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

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


def parse_completion(completion: Any) -> LLMResponse:
    """Turn an OpenAI ChatCompletion-shaped object into a core LLMResponse."""
    choice = completion.choices[0]
    msg = choice.message

    tool_calls: tuple[ToolCall, ...] = ()
    if getattr(msg, "tool_calls", None):
        parsed: list[ToolCall] = []
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
