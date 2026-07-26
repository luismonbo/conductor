"""LLM-judge metric: is the answer fully supported by the retrieved context?
Reuses the fake-tool-call structured-output pattern from the ingestion
plan's LlmNormalizer rather than parsing free-text verdicts."""
from __future__ import annotations

from harness.core.llm.client import LLMClient
from harness.core.types import Message, Role, ToolSpec

from evaluation.harness.metric import MetricResult

_FAITHFULNESS_TOOL = ToolSpec(
    name="score_faithfulness",
    description="Judge whether an answer is fully supported by the given context, with no unsupported claims.",
    parameters={
        "type": "object",
        "properties": {
            "grounded": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["grounded", "reasoning"],
    },
)


class FaithfulnessMetric:
    name = "faithfulness"

    def __init__(self, judge: LLMClient) -> None:
        self._judge = judge

    async def score(self, case, result, tracer) -> MetricResult:
        context = "\n\n".join(sc.chunk.text for sc in result.retrieved) or "(no context retrieved)"
        messages = [
            Message(
                Role.SYSTEM,
                "You judge whether an answer is fully grounded in the given context. "
                "Call score_faithfulness with your verdict.",
            ),
            Message(Role.USER, f"Context:\n{context}\n\nAnswer:\n{result.answer}"),
        ]
        response = await self._judge.generate(messages, tools=[_FAITHFULNESS_TOOL])
        if not response.tool_calls:
            return MetricResult(
                name=self.name, passed=False, score=0.0,
                reason="judge did not return a structured score",
            )
        args = response.tool_calls[0].arguments
        grounded = bool(args["grounded"])
        return MetricResult(
            name=self.name, passed=grounded, score=1.0 if grounded else 0.0,
            reason=args.get("reasoning", ""),
        )
