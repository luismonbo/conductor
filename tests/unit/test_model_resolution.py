"""Precedence: tool pin > agent pin > request override > default.

The UI pick (request_override) steers only components without pins —
a tool pinned to local-gemma is never dragged to claude by the picker.
"""
from harness.core.llm.model_resolution import resolve_model


def test_returns_none_when_nothing_set():
    assert resolve_model() is None


def test_default_used_when_no_pins_or_override():
    assert resolve_model(default="local-gemma") == "local-gemma"


def test_request_override_beats_default():
    assert resolve_model(request_override="claude", default="local-gemma") == "claude"


def test_agent_pin_beats_request_override():
    assert resolve_model(agent_pin="gpt", request_override="claude") == "gpt"


def test_tool_pin_beats_everything():
    assert (
        resolve_model(
            tool_pin="local-gemma",
            agent_pin="gpt",
            request_override="claude",
            default="gemini-flash",
        )
        == "local-gemma"
    )


def test_empty_strings_are_treated_as_unset():
    assert resolve_model(tool_pin="", agent_pin="", request_override="claude") == "claude"
    assert resolve_model(tool_pin="", default="") is None
