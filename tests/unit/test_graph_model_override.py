"""model_override flows from graph state into the LLM call; agent pin beats it."""
import asyncio

from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.orchestration.loop import build_graph


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


def _graph(fake, **kwargs):
    return build_graph(
        fake, MemorySaver(), registry=ToolRegistry(),
        system_prompt="You are a test agent.", **kwargs,
    )


async def test_model_override_reaches_llm():
    fake = FakeLLMClient([LLMResponse(text="hello")])
    graph = _graph(fake)
    await graph.ainvoke(_input_state(model_override="claude"), _config(asyncio.Queue()))
    assert fake.requested_models == ["claude"]


async def test_no_override_passes_none():
    fake = FakeLLMClient([LLMResponse(text="hello")])
    graph = _graph(fake)
    await graph.ainvoke(_input_state(), _config(asyncio.Queue()))
    assert fake.requested_models == [None]


async def test_agent_pin_beats_request_override():
    fake = FakeLLMClient([LLMResponse(text="hello")])
    graph = _graph(fake, model_pin="gpt")
    await graph.ainvoke(_input_state(model_override="claude"), _config(asyncio.Queue()))
    assert fake.requested_models == ["gpt"]
