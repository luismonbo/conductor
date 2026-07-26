"""LLM-judge metric: does the answer actually address the question asked?
Distinct from faithfulness — an answer can be perfectly grounded in context
while failing to address what was asked."""
from __future__ import annotations

from harness.core.llm.client import LLMClient
from harness.core.types import Message, Role, ToolSpec

from evaluation.harness.metric import MetricResult

_RELEVANCY_TOOL = ToolSpec(
    name="score_answer_relevancy",
    description="Judge whether an answer actually addresses the question asked.",
    parameters={
        "type": "object",
        "properties": {
            "relevant": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["relevant", "reasoning"],
    },
)


class AnswerRelevancyMetric:
    name = "answer_relevancy"

    def __init__(self, judge: LLMClient) -> None:
        self._judge = judge

    async def score(self, case, result, tracer) -> MetricResult:
        messages = [
            Message(
                Role.SYSTEM,
                "You judge whether an answer actually addresses the question asked. "
                "Call score_answer_relevancy with your verdict.",
            ),
            Message(Role.USER, f"Question:\n{case.query}\n\nAnswer:\n{result.answer}"),
        ]
        response = await self._judge.generate(messages, tools=[_RELEVANCY_TOOL])
        if not response.tool_calls:
            return MetricResult(
                name=self.name, passed=False, score=0.0,
                reason="judge did not return a structured score",
            )
        args = response.tool_calls[0].arguments
        relevant = bool(args["relevant"])
        return MetricResult(
            name=self.name, passed=relevant, score=1.0 if relevant else 0.0,
            reason=args.get("reasoning", ""),
        )
