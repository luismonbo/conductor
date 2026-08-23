"""Integration smoke test: EvalRunner against a scripted Agent-protocol double.

Scripts _ScriptedAgent to call the calculator tool and produce "108", then
asserts all three metrics pass for the calc_mul_001 case. No network, no real
LLM or agent implementation — fully deterministic.
"""

from dataclasses import dataclass, field

from harness.core.types import AgentResult, AgentState, ToolCall

from evaluation.harness.dataset import Dataset, EvalCase, Expected
from evaluation.harness.runner import EvalRunner
from evaluation.metrics.arg_schema import ArgSchemaMetric
from evaluation.metrics.no_tool_call import NoToolCallMetric
from evaluation.metrics.output_contains import OutputContainsMetric
from evaluation.metrics.tool_call import ToolCallMetric

_DATASETS_DIR = (
    __import__("pathlib").Path(__file__).parent.parent.parent / "evaluation" / "datasets"
)


@dataclass
class _ScriptedAgent:
    """Minimal Agent-protocol double: returns a canned AgentResult and
    replays canned tool-call/tool-result events into the tracer, exactly
    like a real agent run would, without depending on any concrete
    agent implementation."""
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[tuple[str, str, bool]] = field(default_factory=list)  # (name, content, is_error)
    stopped_reason: str = "final_answer"
    tracer: object = None

    async def run(self, state: AgentState) -> AgentResult:
        if self.tracer is not None:
            if self.tool_calls:
                await self.tracer(
                    "llm_response",
                    {"tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls]},
                )
            for name, content, is_error in self.tool_results:
                await self.tracer("tool_result", {"name": name, "content": content, "is_error": is_error})
        return AgentResult(output=self.output, state=state, stopped_reason=self.stopped_reason)


def _make_calc_agent(tracer, memory_seed=None):
    """Build a scripted agent that calls calculator then gives a final answer."""
    return _ScriptedAgent(
        output="The result of 12 * 9 is 108.",
        tool_calls=[ToolCall(id="tc_1", name="calculator", arguments={"expression": "12 * 9"})],
        tool_results=[("calculator", "108", False)],
        tracer=tracer,
    )


def _make_direct_agent(tracer, memory_seed=None):
    """Build a scripted agent that answers without calling any tool."""
    return _ScriptedAgent(output="Hello there!", tracer=tracer)


class TestNoToolCallMetricSmoke:
    def test_direct_case_passes_no_tool_call_metric(self):
        case = EvalCase(
            id="direct_smoke",
            description="smoke",
            input="say hello",
            tags=["smoke"],
            expected=Expected(no_tool_call=True, output_contains=["Hello"]),
        )
        dataset = Dataset([case])
        metrics = [NoToolCallMetric(), OutputContainsMetric()]
        runner = EvalRunner(_make_direct_agent)
        report = runner.run(dataset, metrics, dataset_name="inline")

        assert report.total == 1
        assert report.passed == 1
        case_report = report.cases[0]
        for mr in case_report.metric_results:
            assert mr.passed, f"{mr.name} failed: {mr.reason}"


class TestEvalRunnerSmoke:
    def test_calc_case_passes_all_metrics(self):
        dataset = Dataset.load(_DATASETS_DIR / "tool_use_v1.json").filter_by_tags(
            ["smoke"]
        )
        assert len(dataset.cases) == 1

        metrics = [ToolCallMetric(), ArgSchemaMetric(), OutputContainsMetric()]
        runner = EvalRunner(_make_calc_agent)
        report = runner.run(dataset, metrics, dataset_name="tool_use_v1.json")

        assert report.total == 1
        assert report.passed == 1
        assert report.pass_rate == 1.0

    def test_calc_case_metric_results(self):
        dataset = Dataset.load(_DATASETS_DIR / "tool_use_v1.json").filter_by_tags(
            ["smoke"]
        )
        metrics = [ToolCallMetric(), ArgSchemaMetric(), OutputContainsMetric()]
        runner = EvalRunner(_make_calc_agent)
        report = runner.run(dataset, metrics, dataset_name="tool_use_v1.json")

        case = report.cases[0]
        assert case.case_id == "calc_mul_001"
        for mr in case.metric_results:
            assert mr.passed, f"{mr.name} failed: {mr.reason}"

    def test_report_has_correct_by_metric_counts(self):
        dataset = Dataset.load(_DATASETS_DIR / "tool_use_v1.json").filter_by_tags(
            ["smoke"]
        )
        metrics = [ToolCallMetric(), ArgSchemaMetric(), OutputContainsMetric()]
        runner = EvalRunner(_make_calc_agent)
        report = runner.run(dataset, metrics, dataset_name="tool_use_v1.json")

        by_metric = report._by_metric()
        for name in ("tool_call", "arg_schema", "output_contains"):
            assert by_metric[name]["passed"] == 1
            assert by_metric[name]["failed"] == 0
