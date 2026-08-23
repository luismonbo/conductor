"""Adapts a compiled LangGraph agent graph to the Agent protocol, so
EvalRunner (built around Agent.run(state) -> AgentResult) can drive the
same graph that serves live chat, instead of a separate implementation.

Lives here, not in src/harness/, because this exists purely to make the
graph evaluable — not a product concern.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

from harness.core.types import AgentEvent, AgentResult, AgentState

Tracer = Callable[[str, dict], Awaitable[None]]


class GraphAgentAdapter:
    def __init__(self, graph: Any, tracer: Tracer | None = None) -> None:
        self._graph = graph
        self._trace = tracer

    async def run(self, state: AgentState) -> AgentResult:
        queue: asyncio.Queue = asyncio.Queue()
        stopped_reason_holder: list[str] = ["unknown"]
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "event_queue": queue,
                "stopped_reason_holder": stopped_reason_holder,
            },
        }
        graph_state = {
            "messages": list(state.messages),
            "iteration": 0,
            "max_iterations": state.max_iterations,
        }

        async def _drive() -> dict:
            try:
                return await self._graph.ainvoke(graph_state, config)
            finally:
                await queue.put(None)

        task = asyncio.create_task(_drive())
        events: list[AgentEvent] = []
        while True:
            item = await queue.get()
            if item is None:
                break
            events.append(item)
        result = await task

        if (result or {}).get("__interrupt__"):
            raise RuntimeError(
                "Graph interrupted (tool requires approval) — eval cases "
                "can't exercise requires_approval tools yet."
            )

        if self._trace is not None:
            await self._emit_trace(events)

        final = next((e for e in reversed(events) if e.type in ("final", "error")), None)
        output = final.text if final else ""

        # Build the real post-run state from what ainvoke() actually returned,
        # not the pre-run input — GraphState's messages/iteration are directly
        # compatible with AgentState's own fields.
        result = result or {}
        final_state = AgentState(
            messages=list(result.get("messages", state.messages)),
            max_iterations=state.max_iterations,
            iteration=result.get("iteration", 0),
        )
        return AgentResult(
            output=output, state=final_state, stopped_reason=stopped_reason_holder[0],
        )

    async def _emit_trace(self, events: list[AgentEvent]) -> None:
        for e in events:
            if e.type == "tool_call":
                await self._trace(
                    "llm_response",
                    {"tool_calls": [{"name": e.name, "arguments": e.args}]},
                )
            elif e.type == "tool_result":
                await self._trace(
                    "tool_result",
                    {"name": e.name, "is_error": e.is_error, "content": e.text},
                )
