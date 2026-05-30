"""Offline eval skeleton: run the golden dataset through a built agent and
score tool-selection + answer-substring. Wire this into CI to catch prompt or
model regressions — essential when an unreliable small model is in the loop."""
from __future__ import annotations
import asyncio, json
from pathlib import Path
from harness.config.settings import get_settings
from harness.observability.tracer import TraceCollector
from harness.orchestration.build import build_agent
from harness.core.types import AgentState, Message, Role

async def run() -> None:
    cases = json.loads(Path(__file__).with_name("golden_dataset.json").read_text())["cases"]
    settings = get_settings()
    passed = 0
    for c in cases:
        tracer = TraceCollector()
        agent = build_agent(settings, tracer=tracer)
        res = await agent.run(AgentState(messages=[Message(Role.USER, c["input"])]))
        tools_used = [e[2]["name"] for e in tracer.events if e[1] == "tool_result"]
        ok = True
        if c["expect_tool"] and c["expect_tool"] not in tools_used: ok = False
        if c["expect_substring"] and c["expect_substring"].lower() not in res.output.lower(): ok = False
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']}: tools={tools_used}")
    print(f"\n{passed}/{len(cases)} passed")

if __name__ == "__main__":
    asyncio.run(run())
