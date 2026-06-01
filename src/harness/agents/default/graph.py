"""Default LangGraph agent — six-node StateGraph.

Topology:
  START → call_model
            ├─ wants_tools + iteration_ok → approval_gate
            │     ├─ needs_approval=True  → interrupt() [state saved to checkpointer]
            │     │                         on resume: approved  → execute_tools
            │     │                                   rejected → handle_rejection → call_model (loop)
            │     └─ needs_approval=False → execute_tools → call_model (loop)
            ├─ no_tools               → final
            └─ iteration_limit        → error
  final → END
  error → END

Event delivery: nodes push AgentEvent objects into an asyncio.Queue passed
via config["configurable"]["event_queue"]. The API layer puts the None sentinel
in its _run() finally block to close the SSE stream.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Any, NotRequired
import operator

from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.agents.default.tools import build_registry
from harness.core.llm.client import LLMClient
from harness.core.tools.registry import ToolRegistry
from harness.core.types import AgentEvent, LLMResponse, Message, Role


class GraphState(TypedDict):
    messages: Annotated[list[Message], operator.add]  # append-only reducer
    iteration: int
    max_iterations: int
    decision: NotRequired[dict | None]  # populated on interrupt resume


def build_graph(
    llm: LLMClient,
    checkpointer: Any,
    sub_agents: dict[str, Any] = {},
    registry: ToolRegistry | None = None,
) -> Any:
    """Compile and return the default agent graph.

    `registry` is exposed for testing — tests can inject a registry with
    tools that have requires_approval=True without patching module globals.
    """
    if registry is None:
        registry = build_registry()

    async def _call_model(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        msgs = list(state["messages"])
        if not msgs or msgs[0].role != Role.SYSTEM:
            msgs = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)] + msgs

        response: LLMResponse | None = None
        async for item in llm.stream(msgs, registry.specs()):
            if isinstance(item, str):
                if item:
                    await queue.put(AgentEvent(type="token", text=item))
            else:
                response = item

        if response is None:
            response = LLMResponse(text="")

        for tc in response.tool_calls:
            await queue.put(AgentEvent(
                type="tool_call", name=tc.name, args=tc.arguments, call_id=tc.id,
            ))

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=response.text,
            tool_calls=response.tool_calls,
        )
        return {"messages": [assistant_msg], "iteration": state["iteration"] + 1}

    def _route_after_model(state: GraphState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            return "error"
        last = state["messages"][-1]
        if last.tool_calls:
            return "approval_gate"
        return "final"

    async def _approval_gate(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        last = state["messages"][-1]

        needs_approval = any(
            getattr(registry.get(tc.name), "requires_approval", False)
            for tc in last.tool_calls
        )

        if not needs_approval:
            return {}

        tc_list = [
            {"name": tc.name, "args": tc.arguments, "call_id": tc.id}
            for tc in last.tool_calls
        ]
        await queue.put(AgentEvent(
            type="interrupt",
            args={"mode": "approval", "tool_calls": tc_list},
        ))
        # sentinel is NOT put here — API _run() finally puts it after ainvoke() returns

        decision = interrupt({"mode": "approval", "tool_calls": tc_list})
        return {"decision": decision}

    async def _handle_rejection(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        last = state["messages"][-1]

        # Inject a fake tool-result message for each rejected call so the
        # transcript stays valid (OpenAI format requires tool results for every
        # tool_calls entry in the preceding assistant turn).
        rejection_msgs = [
            Message(
                role=Role.TOOL,
                content="Rejected by user.",
                tool_call_id=tc.id,
                name=tc.name,
            )
            for tc in last.tool_calls
        ]

        await queue.put(AgentEvent(type="thinking", text="[Tool call rejected. Continuing…]"))
        return {"messages": rejection_msgs}

    def _route_after_approval(state: GraphState) -> str:
        decision = state.get("decision")
        if decision is not None:
            return "execute_tools" if decision.get("approved") else "handle_rejection"
        return "execute_tools"

    async def _execute_tools(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        last = state["messages"][-1]
        new_msgs: list[Message] = []

        for tc in last.tool_calls:
            result = await registry.dispatch(tc)
            await queue.put(AgentEvent(
                type="tool_result",
                name=result.name,
                text=result.content,
                is_error=result.is_error,
                call_id=result.tool_call_id,
            ))
            new_msgs.append(Message(
                role=Role.TOOL,
                content=result.content,
                tool_call_id=result.tool_call_id,
                name=result.name,
            ))

        return {"messages": new_msgs}

    async def _final(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        last_assistant = next(
            (m for m in reversed(state["messages"]) if m.role == Role.ASSISTANT),
            None,
        )
        text = last_assistant.content if last_assistant else ""
        await queue.put(AgentEvent(type="final", text=text, stopped_reason="final_answer"))
        return {}

    async def _error(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        await queue.put(AgentEvent(
            type="error",
            text="I couldn't complete this within the allowed reasoning steps.",
        ))
        return {}

    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("call_model", _call_model)
    builder.add_node("approval_gate", _approval_gate)
    builder.add_node("execute_tools", _execute_tools)
    builder.add_node("final", _final)
    builder.add_node("error", _error)
    builder.add_node("handle_rejection", _handle_rejection)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        _route_after_model,
        {"approval_gate": "approval_gate", "final": "final", "error": "error"},
    )
    builder.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {"execute_tools": "execute_tools", "handle_rejection": "handle_rejection"},
    )
    builder.add_edge("handle_rejection", "call_model")
    builder.add_edge("execute_tools", "call_model")
    builder.add_edge("final", END)
    builder.add_edge("error", END)

    return builder.compile(checkpointer=checkpointer)
