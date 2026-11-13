"""The shared agent loop: the seven-node StateGraph every agent runs on.

Topology:
  START → call_model
            ├─ no tool calls            → final
            ├─ tool calls at the limit  → error
            └─ tool calls               → approval_gate
                  ├─ nothing needs approval → execute_tools
                  └─ interrupt() on the turn's next review: each memory proposal,
                     then the other calls together when any of them needs approval
                        ├─ approved, more reviews → approval_gate (next review)
                        ├─ approved, none left    → execute_tools
                        ├─ denied                 → handle_rejection
                        └─ feedback               → refine
  execute_tools, handle_rejection, refine → call_model (the loop)
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
from itertools import takewhile
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
from harness.core.types import AgentEvent, LLMResponse, Message, Role, ToolCall, ToolSpec
from harness.observability.langfuse_tracing import langfuse_configured
from harness.observability.token_accumulator import estimate_tokens


class GraphState(TypedDict):
    messages: Annotated[list[Message], operator.add]  # append-only reducer
    iteration: int
    max_iterations: int
    decision: NotRequired[dict | None]  # the answer to this turn's latest review
    # What a human must approve before this turn's tool calls run (interrupt
    # payloads, in order), and how many of them were approved so far. Worked out
    # once, when the model makes the calls, and stored with them: tools registered
    # differently after a restart must not change what is asked, or which review
    # an answer goes to.
    reviews: NotRequired[list[dict]]
    reviews_approved: NotRequired[int]
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


# "May not have run", not "was not run": a cancel inside execute_tools drops the
# results of calls that had already finished.
_NO_RESULT = (
    "No result was recorded for this tool call: the conversation moved on before it "
    "completed, so it may not have run."
)


def _answer_abandoned_calls(messages: list[Message]) -> list[Message]:
    """`messages` with a result after every tool call that never got one.

    A run can stop between the model's tool calls and their results: a visitor
    writes past a pending approval or cancels, or the step limit ends the turn.
    The checkpoint then holds calls without results, and a provider rejects a
    transcript that has one, so the thread could never continue. The checkpoint
    is append-only, so the result is added to the transcript the model is sent
    and is never stored.
    """
    transcript: list[Message] = []
    unanswered: dict[str, ToolCall] = {}  # the latest assistant turn's calls, by id

    def answer_unanswered() -> None:
        transcript.extend(
            Message(
                role=Role.TOOL, content=_NO_RESULT, tool_call_id=call.id, name=call.name
            )
            for call in unanswered.values()
        )
        unanswered.clear()

    for message in messages:
        if message.role == Role.TOOL:
            unanswered.pop(message.tool_call_id or "", None)
        else:
            # Results directly follow their calls, so any other message ends them.
            answer_unanswered()
        transcript.append(message)
        for call in message.tool_calls:
            unanswered[call.id] = call
    answer_unanswered()
    return transcript


def _window(messages: list[Message], turns: int) -> list[Message]:
    """The messages from the `turns`-th last visitor message on.

    All of them when turns is 0 or the thread is shorter. A cut falls right
    before a visitor message, so a tool call never loses its result. The
    checkpoint keeps everything; this is only what the model is sent.
    """
    if turns <= 0:
        return messages
    starts = [i for i, m in enumerate(messages) if m.role == Role.USER]
    return messages if len(starts) <= turns else messages[starts[-turns]:]


_NODE_FOR_ACTION = {
    "approve": "execute_tools",
    "feedback": "refine",
    "deny": "handle_rejection",
}

# The result each call of a declined turn is given.
_REJECTED = "Rejected by user."

# Sent after those results to the model call that answers the denial, as a system
# message, and never stored. Results alone did not do it: told "Rejected by user."
# (or LangChain's wording), gpt-5.4-mini answered a declined memory with "I'll
# remember that", as if it had been saved; models weigh tool output below system
# messages. Without "Tell them it was not done" it still did. Some local chat
# templates (Qwen3's) refuse a system message that is not the first.
_DECLINED_NOTICE = (
    "The user was asked to approve what you tried to do above and declined, so it "
    "did not run and nothing was done. Tell them it was not done, and don't say or "
    "suggest that it was or will be."
)


def _ends_with_a_denial(messages: list[Message]) -> bool:
    """Whether `messages` end with the results the calls of a declined turn get."""
    results = list(takewhile(lambda message: message.role == Role.TOOL, reversed(messages)))
    return bool(results) and all(message.content == _REJECTED for message in results)


def _decided_action(decision: dict | None) -> str:
    """What an approval answer decides: "approve", "feedback" or "deny".

    None means nothing needed approval. The answer is whatever JSON the client
    sent, so anything that is not clearly an approval or feedback is a denial.
    """
    if decision is None:
        return "approve"
    action = decision.get("action")
    if isinstance(action, str) and action in ("deny", "feedback"):
        return action
    # An approval is `action: "approve"`, the legacy `approved: true`, or both.
    # Each key that is present must say so: only a real true counts (the string
    # "false" is truthy), and two keys that disagree are not an approval.
    answered = "action" in decision or "approved" in decision
    action_approves = "action" not in decision or action == "approve"
    legacy_approves = "approved" not in decision or decision["approved"] is True
    return "approve" if answered and action_approves and legacy_approves else "deny"


async def _stream_results(queue: asyncio.Queue, messages: list[Message]) -> None:
    """Stream tool messages written in place of a run, as _execute_tools streams one.

    The browser then shows each call's outcome as that call's result. A notice
    streamed as text ran into the model's reply, which the UI showed twice.
    """
    for message in messages:
        await queue.put(AgentEvent(
            type="tool_result",
            name=message.name or "",
            text=message.content,
            call_id=message.tool_call_id or "",
        ))


def build_graph(
    llm: LLMClient,
    checkpointer: Any,
    *,
    registry: ToolRegistry,
    system_prompt: str,
    model_pin: str | None = None,
    context_turns: int = 0,
) -> Any:
    """Compile the loop for one agent: its tools, its prompt, its optional model pin.

    orchestration.build passes these from the agent's AgentDefinition, and
    context_turns from the settings (0 sends the whole thread); tests pass
    them directly.
    """

    async def _call_model(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        msgs = _answer_abandoned_calls(_window(state["messages"], context_turns))
        if not msgs or msgs[0].role != Role.SYSTEM:
            msgs = [Message(role=Role.SYSTEM, content=system_prompt)] + msgs
        if _ends_with_a_denial(msgs):
            msgs = [*msgs, Message(role=Role.SYSTEM, content=_DECLINED_NOTICE)]

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
        return {
            "messages": [assistant_msg],
            "iteration": state["iteration"] + 1,
            # Stored with the calls they cover (see GraphState). Approvals start
            # at none for every turn, even when the turn before it was abandoned
            # halfway through its reviews.
            "reviews": _reviews(response.tool_calls),
            "reviews_approved": 0,
        }

    def _route_after_model(state: GraphState) -> str:
        # An answer ends the run whatever the step count: the limit stops a model
        # that keeps asking for tools, and must not discard an answer it gave.
        last = state["messages"][-1]
        if not last.tool_calls:
            return "final"
        if state["iteration"] >= state["max_iterations"]:
            return "error"
        return "approval_gate"

    def _reviews(calls: tuple[ToolCall, ...]) -> list[dict]:
        """What a human must approve before `calls` run: interrupt payloads, in order.

        Each memory proposal is its own review, so the UI can render approve/deny
        buttons and a free-text feedback input for it. The other calls share one
        review when any of them needs approval.
        """
        # Keyed on the call name, so a hallucinated `remember` would count even
        # when the tool is not registered. Require registration too, otherwise a
        # public deployment can be made to pause on a tool that cannot execute.
        proposes_memory = registry.get("remember") is not None
        proposals = [tc for tc in calls if proposes_memory and tc.name == "remember"]
        others = [tc for tc in calls if not (proposes_memory and tc.name == "remember")]

        reviews: list[dict] = [
            {
                "mode": "memory_proposal",
                "proposed": tc.arguments.get("text", ""),
                "call_id": tc.id,
            }
            for tc in proposals
        ]
        needs_approval = any(
            getattr(registry.get(tc.name), "requires_approval", False) for tc in others
        )
        if needs_approval:
            listed = [
                {"name": tc.name, "args": tc.arguments, "call_id": tc.id} for tc in others
            ]
            reviews.append({"mode": "approval", "tool_calls": listed})
        return reviews

    def _pending_reviews(state: GraphState) -> list[dict]:
        """This turn's reviews that no human has approved yet."""
        reviews = state.get("reviews")
        if reviews is None:
            # A thread saved before reviews were stored, perhaps paused on one:
            # work them out as that version did, from the tools registered now.
            reviews = _reviews(state["messages"][-1].tool_calls)
        return reviews[state.get("reviews_approved") or 0:]

    async def _approval_gate(state: GraphState, config: RunnableConfig) -> dict:
        # NOTE: queue.put() is intentionally absent here — interrupt events are
        # emitted by _run_graph after ainvoke() returns so resume reruns of this
        # node don't fire a duplicate event into the new SSE stream.
        pending = _pending_reviews(state)
        if not pending:
            # Written even so: an answer left in the checkpoint from an earlier turn
            # would decide this one, and one "deny" would then reject every later
            # tool call.
            return {"decision": None}
        # One interrupt() per invocation, as LangGraph's docs ask: a resume reruns
        # this node from the top and matches answers to interrupt() calls by
        # position. The review comes from the checkpoint, so the rerun asks about
        # the one the human answered. The turn's next review gets an invocation of
        # its own (_route_after_approval).
        decision = interrupt(pending[0])
        if _decided_action(decision) != "approve":
            return {"decision": decision}
        approved = (state.get("reviews_approved") or 0) + 1
        return {"decision": decision, "reviews_approved": approved}

    async def _handle_rejection(state: GraphState, config: RunnableConfig) -> dict:
        queue: asyncio.Queue = config["configurable"]["event_queue"]
        # Loop is bounded: _call_model increments iteration; _route_after_model enforces max_iterations.
        last = state["messages"][-1]

        # Inject a fake tool-result message for each rejected call so the
        # transcript stays valid (OpenAI format requires tool results for every
        # tool_calls entry in the preceding assistant turn). _call_model follows
        # them with _DECLINED_NOTICE for the reply.
        rejection_msgs = [
            Message(
                role=Role.TOOL,
                content=_REJECTED,
                tool_call_id=tc.id,
                name=tc.name,
            )
            for tc in last.tool_calls
        ]

        await _stream_results(queue, rejection_msgs)
        return {"messages": rejection_msgs}

    def _route_after_approval(state: GraphState) -> str:
        # The first answer that is not an approval decides the whole turn, so no
        # call in it runs unless every review was approved.
        action = _decided_action(state.get("decision"))
        if action == "approve" and _pending_reviews(state):
            return "approval_gate"
        return _NODE_FOR_ACTION[action]

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

        await _stream_results(queue, refine_msgs)
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
            "approval_gate": "approval_gate",
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
