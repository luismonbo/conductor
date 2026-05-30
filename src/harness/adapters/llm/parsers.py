"""Tool-call parser implementations.

NativeToolCallParser   - trusts the model/SDK to emit structured tool_calls.
                         The adapter has already parsed them, so extract() just
                         passes text through. Used with Azure OpenAI and hosted
                         Gemma-4 26B/31B.

PromptedToolCallParser - for small local models (Gemma-4 E2B on a Pi) that
                         can't be trusted to emit structured calls. Injects a
                         strict fenced-JSON format into the system prompt and
                         tolerantly extracts calls from raw text.

Both satisfy core.llm.tool_parsing.ToolCallParser and must pass the same
contract suite, so you can A/B them against one model and measure robustness
before committing the Pi build.
"""
from __future__ import annotations

import json
import re
import uuid

from harness.core.types import ToolCall, ToolSpec

_TOOL_CALL_FENCE = re.compile(
    r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL
)


class NativeToolCallParser:
    def system_prompt_addendum(self, tools: list[ToolSpec]) -> str:
        return ""

    def extract(self, raw_text: str) -> tuple[str, list[ToolCall]]:
        # Native path: structured calls handled in the adapter. Text is final.
        return raw_text, []


class PromptedToolCallParser:
    """Use only when the backend cannot do reliable native tool-calling."""

    def system_prompt_addendum(self, tools: list[ToolSpec]) -> str:
        lines = [
            "You can call tools. To call one, output a fenced block EXACTLY:",
            "```tool_call",
            '{"name": "<tool_name>", "arguments": {<json args>}}',
            "```",
            "Call at most one tool per turn. If you can answer directly, do so",
            "with plain text and no fenced block. Available tools:",
        ]
        for t in tools:
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  params schema: {json.dumps(t.parameters)}")
        return "\n".join(lines)

    def extract(self, raw_text: str) -> tuple[str, list[ToolCall]]:
        calls: list[ToolCall] = []
        for match in _TOOL_CALL_FENCE.finditer(raw_text):
            try:
                obj = json.loads(match.group(1))
                name = obj["name"]
                args = obj.get("arguments", {})
                if isinstance(args, str):  # small models sometimes double-encode
                    args = json.loads(args)
                calls.append(
                    ToolCall(id=str(uuid.uuid4()), name=name, arguments=args)
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                # Malformed call: skip it. The agent will see no tool call and
                # the model gets another turn — better than crashing.
                continue
        cleaned = _TOOL_CALL_FENCE.sub("", raw_text).strip()
        return cleaned, calls
