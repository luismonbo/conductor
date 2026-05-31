"""AgentEvent unit tests."""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from harness.core.types import AgentEvent
from harness.api.schemas import AgentEventDTO


@pytest.mark.unit
def test_agent_event_is_frozen():
    """AgentEvent is immutable; attempting mutation raises FrozenInstanceError."""
    event = AgentEvent(type="thinking", text="hello")
    with pytest.raises(FrozenInstanceError):
        event.text = "goodbye"


@pytest.mark.unit
def test_agent_event_thinking_defaults():
    """AgentEvent with minimal args provides expected defaults."""
    event = AgentEvent(type="thinking", text="hello")
    assert event.type == "thinking"
    assert event.text == "hello"
    assert event.name == ""
    assert event.args == {}
    assert event.call_id == ""
    assert event.is_error is False
    assert event.stopped_reason == ""


@pytest.mark.unit
def test_agent_event_tool_call_with_args():
    """AgentEvent tool_call stores name and args correctly."""
    event = AgentEvent(type="tool_call", name="calc", args={"x": 1, "y": 2}, call_id="call-123")
    assert event.type == "tool_call"
    assert event.name == "calc"
    assert event.args == {"x": 1, "y": 2}
    assert event.call_id == "call-123"


@pytest.mark.unit
def test_agent_event_dto_serialization():
    """AgentEventDTO.model_dump() includes all fields."""
    dto = AgentEventDTO(type="done", text="42", stopped_reason="final_answer")
    dumped = dto.model_dump()
    assert dumped["type"] == "done"
    assert dumped["text"] == "42"
    assert dumped["stopped_reason"] == "final_answer"
    assert dumped["name"] == ""
    assert dumped["args"] == {}
    assert dumped["call_id"] == ""
    assert dumped["is_error"] is False
