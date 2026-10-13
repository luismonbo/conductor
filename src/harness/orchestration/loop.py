"""The shared agent loop: the seven-node StateGraph every agent runs on.

Topology:
  START → call_model
            ├─ wants_tools + iteration_ok → approval_gate
            │     ├─ memory tool    → interrupt() → approve → execute_tools → call_model (loop)
            │     │                               → deny    → handle_rejection → call_model (loop)
            │     │                               → feedback → refine → call_model (loop)
            │     ├─ other approval → interrupt() → approve → execute_tools → call_model (loop)
            │     │                               → deny    → handle_rejection → call_model (loop)
            │     │                               → feedback → refine → call_model (loop)
            │     └─ no approval    → execute_tools → call_model (loop)
            ├─ no_tools             → final
            └─ iteration_limit      → error
  final → END
  error → END

Event delivery: nodes push AgentEvent objects into an asyncio.Queue passed
via config["configurable"]["event_queue"]. The API layer puts the None sentinel
in its _run() finally block to close the SSE stream.

Each agent supplies its prompt, tools and optional model pin through its
AgentDefinition (harness.agents.definition); orchestration.build.build_agent
compiles it here.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, NotRequired
import operator

from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from harness.core.llm.client import LLMClient
from harness.core.llm.errors import ContentFiltered
from harness.core.llm.model_resolution import resolve_model
from harness.core.tools.registry import ToolRegistry
from harness.core.types import AgentEvent, LLMResponse, Message, Role, ToolSpec
from harness.observability.langfuse_tracing import langfuse_configured
from harness.observability.token_accumulator import estimate_tokens


class GraphState(TypedDict):
    messages: Annotated[list[Message], operator.add]  # append-only reducer
    iteration: int
    max_iterations: int
    decision: NotRequired[dict | None]  # populated on interrupt resume
    model_override: NotRequired[str | None]  # UI picker; pins beat it


def _prompt_chars(msgs: list[Message], specs: list[ToolSpec]) -> int:
    """Characters one model call sends: every message plus the tool definitions."""
    message_chars = sum(
        len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls) for m in msgs
    )
    return message_chars + sum(len(repr(spec)) for spec in specs)


def _estimated_usage(
    msgs: list[Message], specs: list[ToolSpec], streamed: list[str]
) -> tuple[int, int]:
    """(input, output) tokens for a call the provider never reported usage for."""
    return (
        estimate_tokens(_prompt_chars(msgs, specs)),
        estimate_tokens(sum(len(t) for t in streamed)),
    )


def build_graph(
    llm: LLMClient,
    checkpointer: Any,
    *,
    registry: ToolRegistry,
    system_prompt: str,
    model_pin: str | None = None,
) -> Any:
    """Compile the loop for one agent: its tools, its prompt, its optional model pin.

    orchestration.build passes these from the agent's AgentDefinition; tests
    pass them directly.
    """

    async def _call_model(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        msgs = list(state["messages"])
        if not msgs or msgs[0].role != Role.SYSTEM:
            msgs = [Message(role=Role.SYSTEM, content=system_prompt)] + msgs

        model = resolve_model(
            agent_pin=model_pin,
            request_override=state.get("model_override"),
        )
        specs = registry.specs()
        response: LLMResponse | None = None
        streamed: list[str] = []
        accumulator = config["configurable"].get("token_accumulator")
        try:
            async for item in llm.stream(msgs, specs, model=model):
                if isinstance(item, str):
                    if item:
                        streamed.append(item)
                        await queue.put(AgentEvent(type="token", text=item))
                else:
                    response = item
        except asyncio.CancelledError:
            # A call cancelled mid-stream (the visitor pressed stop or left) is
            # still billed, but the provider never reports its usage. Record an
            # estimate, or the daily token budget never sees cancelled calls.
            if accumulator:
                input_t, output_t = _estimated_usage(msgs, specs, streamed)
                accumulator.add(input_tokens=input_t, output_tokens=output_t)
            raise

        if response is None:
            response = LLMResponse(text="")

        filtered = response.finish_reason == "content_filter"
        if accumulator:
            input_t, output_t = response.token_usage
            if filtered and not (input_t or output_t):
                # A filtered stream may end without a usage report; the call was
                # still billed, so estimate it as a cancelled call is.
                input_t, output_t = _estimated_usage(msgs, specs, streamed)
            accumulator.add(input_tokens=input_t, output_tokens=output_t)

        if filtered:
            # The provider's filter cut the answer off. Raising keeps the partial
            # answer out of the checkpoint; the API turns this into the visitor's
            # content-filter sentence (security/egress.py).
            raise ContentFiltered("completion")

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
        last = state["messages"][-1]

        # Memory proposals get their own interrupt shape so the UI can render
        # approve/deny buttons + a free-text feedback input.
        # NOTE: queue.put() is intentionally absent here — interrupt events are
        # emitted by _run_graph after ainvoke() returns so resume reruns of this
        # node don't fire a duplicate event into the new SSE stream.
        # Keyed on the call name, so a hallucinated `remember` would reach this
        # branch even when the tool is not registered. Require registration too,
        # otherwise a public deployment can be made to pause on a tool that
        # cannot execute.
        memory_calls = [
            tc for tc in last.tool_calls
            if tc.name == "remember" and registry.get(tc.name) is not None
        ]
        if memory_calls:
            tc = memory_calls[0]
            proposed = tc.arguments.get("text", "")
            payload = {"mode": "memory_proposal", "proposed": proposed, "call_id": tc.id}
            decision = interrupt(payload)
            return {"decision": decision}

        # Regular tool approval (any other tool with requires_approval=True)
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
        decision = interrupt({"mode": "approval", "tool_calls": tc_list})
        return {"decision": decision}

    async def _handle_rejection(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        # Loop is bounded: _call_model increments iteration; _route_after_model enforces max_iterations.
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
        if decision is None:
            return "execute_tools"
        action = decision.get("action")
        if action == "approve":
            return "execute_tools"
        if action == "feedback":
            return "refine"
        if action == "deny":
            return "handle_rejection"
        # Legacy tool-approval shape: {"approved": true/false}
        return "execute_tools" if decision.get("approved") else "handle_rejection"

    async def _refine(state: GraphState, config: RunnableConfig) -> dict:
        """Handle user feedback on any tool that requires approval.

        Injects a tool result carrying the feedback so the agent can revise
        its approach and retry. Works for any requires_approval tool, not just
        memory writes.
        """
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        last = state["messages"][-1]
        feedback = (state.get("decision") or {}).get("feedback", "")

        refine_msgs = []
        for tc in last.tool_calls:
            refine_msgs.append(Message(
                role=Role.TOOL,
                content=(
                    f"User feedback on {tc.name}: {feedback}. "
                    "Please revise your approach and try again."
                ),
                tool_call_id=tc.id,
                name=tc.name,
            ))

        await queue.put(AgentEvent(type="thinking", text=f"[User feedback: {feedback}]"))
        return {"messages": refine_msgs, "decision": None}

    async def _execute_tools(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        last = state["messages"][-1]
        new_msgs: list[Message] = []

        for tc in last.tool_calls:
            # Dispatch first, unconditionally — tool execution must never
            # depend on Langfuse succeeding. registry.dispatch() calls our own
            # Tool protocol directly, not a LangChain BaseTool, so the graph's
            # CallbackHandler has no visibility into it (same reason the LLM
            # call needs langfuse.openai's patch); the span below is a
            # best-effort add-on, not a gate on execution.
            result = await registry.dispatch(tc)
            if langfuse_configured():
                try:
                    from langfuse import get_client

                    with get_client().start_as_current_observation(
                        as_type="tool", name=tc.name
                    ) as obs:
                        obs.update(input=tc.arguments, output=result.content)
                except Exception:
                    logging.getLogger(__name__).exception("langfuse tool span failed")
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
        if holder := config["configurable"].get("stopped_reason_holder"):
            holder[0] = "final_answer"
        return {}

    async def _error(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        await queue.put(AgentEvent(
            type="error",
            text="I couldn't complete this within the allowed reasoning steps.",
        ))
        if holder := config["configurable"].get("stopped_reason_holder"):
            holder[0] = "max_iterations"
        return {}

    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("call_model", _call_model)
    builder.add_node("approval_gate", _approval_gate)
    builder.add_node("execute_tools", _execute_tools)
    builder.add_node("final", _final)
    builder.add_node("error", _error)
    builder.add_node("handle_rejection", _handle_rejection)
    builder.add_node("refine", _refine)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        _route_after_model,
        {"approval_gate": "approval_gate", "final": "final", "error": "error"},
    )
    builder.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {
            "execute_tools": "execute_tools",
            "handle_rejection": "handle_rejection",
            "refine": "refine",
        },
    )
    builder.add_edge("handle_rejection", "call_model")
    builder.add_edge("execute_tools", "call_model")
    builder.add_edge("refine", "call_model")
    builder.add_edge("final", END)
    builder.add_edge("error", END)

    return builder.compile(checkpointer=checkpointer)
