"""ReAct agent: the observe -> think -> act loop.

Model-agnostic and framework-agnostic. It composes an LLMClient and a
ToolRegistry (both Protocols) and nothing else. Swapping Azure for Gemma-on-Pi
or LangGraph for anything else does not touch this file — that is the test of
whether the seams are real.

Design notes:
  * max_iterations is enforced as a FAILURE boundary. A small model that loops
    without ever producing a final answer hits the ceiling and we return
    stopped_reason="max_iterations" rather than pretending success.
  * Tool errors are fed back to the model as observations, giving it a chance
    to recover (e.g. retry with corrected arguments) instead of aborting.
  * Every iteration is handed to an optional tracer so observability sees each
    think/act step — the silent-loop failure mode is otherwise invisible.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from harness.core.llm.client import LLMClient
from harness.core.tools.registry import ToolRegistry
from harness.core.types import (
    AgentResult,
    AgentState,
    Message,
    Role,
)

# A tracer is just an async callback; the real one lives in observability/.
Tracer = Callable[[str, dict], Awaitable[None]]


async def _noop_tracer(event: str, data: dict) -> None:  # pragma: no cover
    return None


class ReActAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._trace = tracer or _noop_tracer

    async def run(self, state: AgentState) -> AgentResult:
        # Ensure a system message leads the transcript exactly once.
        if not state.messages or state.messages[0].role != Role.SYSTEM:
            state.messages.insert(
                0, Message(role=Role.SYSTEM, content=self._system_prompt)
            )

        tool_specs = self._tools.specs()

        while state.iteration < state.max_iterations:
            state.iteration += 1
            await self._trace("iteration_start", {"n": state.iteration})

            response = await self._llm.generate(state.messages, tool_specs)
            await self._trace(
                "llm_response",
                {
                    "text": response.text,
                    "tool_calls": [c.name for c in response.tool_calls],
                    "usage": response.usage,
                },
            )

            # Record the assistant turn (text + any tool requests).
            state.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            # No tools requested -> this is the final answer.
            if not response.wants_tools:
                return AgentResult(
                    output=response.text,
                    state=state,
                    stopped_reason="final_answer",
                )

            # Act: dispatch each requested tool and feed observations back.
            for call in response.tool_calls:
                result = await self._tools.dispatch(call)
                await self._trace(
                    "tool_result",
                    {"name": result.name, "is_error": result.is_error},
                )
                state.messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                        name=result.name,
                    )
                )

        # Loop ceiling hit without a final answer: explicit failure.
        await self._trace("max_iterations", {"limit": state.max_iterations})
        return AgentResult(
            output=(
                "I couldn't complete this within the allowed reasoning steps."
            ),
            state=state,
            stopped_reason="max_iterations",
        )
