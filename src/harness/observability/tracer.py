"""Minimal tracer + cost tracker.

The ReAct loop emits events (iteration_start, llm_response, tool_result,
max_iterations) to a tracer callback. This collector captures them per run so
you can see the silent-loop failure mode and tally token cost. In production
you'd forward these to OpenTelemetry / Azure Monitor; the interface stays the
same.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, AsyncGenerator

from harness.core.types import AgentEvent


@dataclass
class TraceCollector:
    events: list[tuple[float, str, dict]] = field(default_factory=list)
    _t0: float = field(default_factory=perf_counter)

    async def __call__(self, event: str, data: dict[str, Any]) -> None:
        self.events.append((perf_counter() - self._t0, event, data))

    @property
    def total_tokens(self) -> int:
        return sum(
            d.get("usage", {}).get("total_tokens", 0)
            for _, e, d in self.events
            if e == "llm_response"
        )

    @property
    def iterations(self) -> int:
        return sum(1 for _, e, _ in self.events if e == "iteration_start")

    def summary(self) -> dict:
        return {
            "iterations": self.iterations,
            "total_tokens": self.total_tokens,
            "tool_calls": sum(1 for _, e, _ in self.events if e == "tool_result"),
            "tool_errors": sum(
                1 for _, e, d in self.events if e == "tool_result" and d.get("is_error")
            ),
        }


class StreamingTracer:
    """Queue-backed tracer that converts raw loop events into AgentEvent objects.

    Usage pattern (in the SSE endpoint):
        tracer = StreamingTracer()
        task = asyncio.create_task(agent.run(state))
        # In run_agent coroutine, call tracer.finish(done_event) at the end.
        async for event in tracer.drain():
            yield sse_line(event)
        await task
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def __call__(self, event: str, data: dict[str, Any]) -> None:
        """Called by the ReAct loop at each step; enqueues AgentEvent objects."""
        if event == "llm_response":
            if data.get("text"):
                await self._queue.put(AgentEvent(type="thinking", text=data["text"]))
            for tc in data.get("tool_calls", []):
                await self._queue.put(AgentEvent(
                    type="tool_call",
                    name=tc["name"],
                    args=tc.get("arguments", {}),
                    call_id=tc.get("id", ""),
                ))
        elif event == "tool_result":
            await self._queue.put(AgentEvent(
                type="tool_result",
                name=data["name"],
                is_error=data.get("is_error", False),
                text=data.get("content", ""),
            ))
        # iteration_start and max_iterations are internal; not exposed to client

    async def finish(self, event: AgentEvent) -> None:
        """Enqueue the terminal event (done or error) and close the stream."""
        await self._queue.put(event)
        await self._queue.put(None)  # sentinel

    async def drain(self) -> AsyncGenerator[AgentEvent, None]:
        """Yield AgentEvents until the sentinel None is received."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item
