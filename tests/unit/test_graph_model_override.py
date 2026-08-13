"""model_override flows from graph state into the LLM call; agent pin beats it."""
import asyncio

from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.agents.default.graph import build_graph
from harness.core.types import LLMResponse, Message, Role


def _input_state(model_override=None):
    state = {
        "messages": [Message(role=Role.USER, content="hi")],
        "iteration": 0,
        "max_iterations": 3,
    }
    if model_override is not None:
        state["model_override"] = model_override
    return state


def _config(queue):
    return {"configurable": {"thread_id": "t1", "event_queue": queue}}


async def test_model_override_reaches_llm():
    fake = FakeLLMClient([LLMResponse(text="hello")])
    graph = build_graph(fake, MemorySaver())
    await graph.ainvoke(_input_state(model_override="claude"), _config(asyncio.Queue()))
    assert fake.requested_models == ["claude"]


async def test_no_override_passes_none():
    fake = FakeLLMClient([LLMResponse(text="hello")])
    graph = build_graph(fake, MemorySaver())
    await graph.ainvoke(_input_state(), _config(asyncio.Queue()))
    assert fake.requested_models == [None]


async def test_agent_pin_beats_request_override():
    fake = FakeLLMClient([LLMResponse(text="hello")])
    graph = build_graph(fake, MemorySaver(), model_pin="gpt")
    await graph.ainvoke(_input_state(model_override="claude"), _config(asyncio.Queue()))
    assert fake.requested_models == ["gpt"]
