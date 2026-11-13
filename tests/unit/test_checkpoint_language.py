"""The run's language rides in config["configurable"] without reaching a checkpoint."""
from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import MemorySaver

from harness.adapters.llm.fake import FakeLLMClient
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, Message, Role
from harness.orchestration.loop import build_graph
from harness.security.visitor_text import VisitorLanguage


async def test_the_runs_language_never_reaches_a_checkpoint():
    saver = MemorySaver()
    graph = build_graph(
        FakeLLMClient([LLMResponse(text="ok")]), saver,
        registry=ToolRegistry(), system_prompt="p",
    )
    config = {"configurable": {
        "thread_id": "t", "event_queue": asyncio.Queue(),
        "visitor_language": VisitorLanguage("it"), "probe": "plain",
    }}

    await graph.ainvoke(
        {"messages": [Message(role=Role.USER, content="ciao")], "iteration": 0,
         "max_iterations": 2},
        config,
    )

    metadata = [c.metadata async for c in saver.alist({"configurable": {"thread_id": "t"}})]
    # The trap is real: LangGraph copies a plain string from configurable into every
    # checkpoint's metadata, which is why the language travels wrapped.
    assert metadata and all(m.get("probe") == "plain" for m in metadata)
    assert all("visitor_language" not in m for m in metadata)
    assert all("it" not in m.values() for m in metadata)
