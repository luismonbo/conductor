"""Contract suite for ToolCallParser. Runs against BOTH parsers so you can
verify the prompted parser recovers tool calls that the native path gets from
the SDK. This is where you'd add adversarial fixtures (malformed JSON, prose
calls) before trusting a small model on the Pi."""
from __future__ import annotations

import pytest

from harness.adapters.llm.parsers import NativeToolCallParser, PromptedToolCallParser
from harness.core.types import ToolSpec

TOOLS = [
    ToolSpec(
        name="calculator",
        description="do math",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
    )
]


def test_native_passes_text_through():
    parser = NativeToolCallParser()
    assert parser.system_prompt_addendum(TOOLS) == ""
    text, calls = parser.extract("just a final answer")
    assert text == "just a final answer"
    assert calls == []


def test_prompted_injects_format():
    parser = PromptedToolCallParser()
    addendum = parser.system_prompt_addendum(TOOLS)
    assert "tool_call" in addendum
    assert "calculator" in addendum


def test_prompted_extracts_well_formed_call():
    parser = PromptedToolCallParser()
    raw = (
        "I'll compute that.\n"
        '```tool_call\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n```'
    )
    text, calls = parser.extract(raw)
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].arguments["expression"] == "2+2"
    assert "tool_call" not in text  # fence stripped from visible text


def test_prompted_skips_malformed_call():
    parser = PromptedToolCallParser()
    raw = '```tool_call\n{"name": "calculator", broken json}\n```'
    text, calls = parser.extract(raw)
    assert calls == []  # malformed -> skipped, not crashed
