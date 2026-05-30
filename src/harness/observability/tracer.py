"""Minimal tracer + cost tracker.

The ReAct loop emits events (iteration_start, llm_response, tool_result,
max_iterations) to a tracer callback. This collector captures them per run so
you can see the silent-loop failure mode and tally token cost. In production
you'd forward these to OpenTelemetry / Azure Monitor; the interface stays the
same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class TraceCollector:
    events: list[tuple[float, str, dict]] = field(default_factory=list)
    _t0: float = field(default_factory=perf_counter)

    async def __call__(self, event: str, data: dict) -> None:
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
