"""Tests for StreamingTracer in harness.observability.tracer."""
from __future__ import annotations

from harness.core.types import AgentEvent
from harness.observability.tracer import StreamingTracer


async def drain_all(tracer: StreamingTracer) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for e in tracer.drain():
        events.append(e)
    return events


class TestStreamingTracerLlmResponse:
    async def test_text_enqueues_thinking_event(self) -> None:
        tracer = StreamingTracer()
        await tracer("llm_response", {"text": "Hello world", "tool_calls": []})
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        # Last event is the done sentinel from finish(); first should be thinking.
        thinking = [e for e in events if e.type == "thinking"]
        assert len(thinking) == 1
        assert thinking[0].text == "Hello world"

    async def test_tool_calls_enqueue_tool_call_events(self) -> None:
        tracer = StreamingTracer()
        await tracer(
            "llm_response",
            {
                "text": "",
                "tool_calls": [
                    {"name": "calculator", "arguments": {"op": "add", "a": 1, "b": 2}, "id": "tc-001"},
                    {"name": "search", "arguments": {"query": "python"}, "id": "tc-002"},
                ],
            },
        )
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        tool_calls = [e for e in events if e.type == "tool_call"]
        assert len(tool_calls) == 2

        first = tool_calls[0]
        assert first.name == "calculator"
        assert first.args == {"op": "add", "a": 1, "b": 2}
        assert first.call_id == "tc-001"

        second = tool_calls[1]
        assert second.name == "search"
        assert second.args == {"query": "python"}
        assert second.call_id == "tc-002"

    async def test_empty_text_and_no_tool_calls_enqueues_nothing(self) -> None:
        tracer = StreamingTracer()
        await tracer("llm_response", {"text": "", "tool_calls": []})
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        # Only the done event from finish() should be present.
        assert len(events) == 1
        assert events[0].type == "done"


class TestStreamingTracerToolResult:
    async def test_tool_result_event_fields(self) -> None:
        tracer = StreamingTracer()
        await tracer(
            "tool_result",
            {"name": "calculator", "is_error": False, "content": "42"},
        )
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        tool_results = [e for e in events if e.type == "tool_result"]
        assert len(tool_results) == 1
        result = tool_results[0]
        assert result.name == "calculator"
        assert result.is_error is False
        assert result.text == "42"

    async def test_tool_result_error_flag(self) -> None:
        tracer = StreamingTracer()
        await tracer(
            "tool_result",
            {"name": "bad_tool", "is_error": True, "content": "something went wrong"},
        )
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        tool_results = [e for e in events if e.type == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True


class TestStreamingTracerInternalEvents:
    async def test_iteration_start_enqueues_nothing(self) -> None:
        tracer = StreamingTracer()
        await tracer("iteration_start", {"n": 1})
        await tracer("iteration_start", {"n": 2})
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        # Only the done terminal event should appear.
        assert len(events) == 1
        assert events[0].type == "done"

    async def test_max_iterations_enqueues_nothing(self) -> None:
        tracer = StreamingTracer()
        await tracer("max_iterations", {"limit": 8})
        await tracer.finish(AgentEvent(type="done"))

        events = await drain_all(tracer)
        assert len(events) == 1
        assert events[0].type == "done"


class TestStreamingTracerFinish:
    async def test_done_event_is_last_before_drain_stops(self) -> None:
        tracer = StreamingTracer()
        await tracer("llm_response", {"text": "thinking...", "tool_calls": []})
        done_event = AgentEvent(type="done", text="Final answer", stopped_reason="final_answer")
        await tracer.finish(done_event)

        events = await drain_all(tracer)
        assert events[-1] == done_event
        assert events[-1].type == "done"
        assert events[-1].stopped_reason == "final_answer"

    async def test_error_event_propagated_correctly(self) -> None:
        tracer = StreamingTracer()
        error_event = AgentEvent(type="error", text="Something failed", is_error=True)
        await tracer.finish(error_event)

        events = await drain_all(tracer)
        assert len(events) == 1
        assert events[0] == error_event
        assert events[0].is_error is True
