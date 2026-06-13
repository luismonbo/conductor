"""Integration smoke test: EvalRunner with FakeLLMClient.

Scripts the fake client to call the calculator tool and produce "108",
then asserts all three metrics pass for the calc_mul_001 case.
No network, no real LLM — fully deterministic.
"""
import pytest

from harness.adapters.llm.fake import FakeLLMClient
from harness.adapters.memory.in_memory import InMemoryLongTerm
from harness.adapters.tools.calculator import CalculatorTool
from harness.adapters.tools.recall import RecallTool
from harness.core.agents.react import ReActAgent
from harness.core.tools.registry import ToolRegistry
from harness.core.types import LLMResponse, ToolCall

from evaluation.harness.dataset import Dataset, EvalCase, Expected
from evaluation.harness.runner import EvalRunner
from evaluation.metrics.arg_schema import ArgSchemaMetric
from evaluation.metrics.no_tool_call import NoToolCallMetric
from evaluation.metrics.output_contains import OutputContainsMetric
from evaluation.metrics.tool_call import ToolCallMetric

_DATASETS_DIR = (
    __import__("pathlib").Path(__file__).parent.parent.parent / "evaluation" / "datasets"
)


def _make_calc_agent(tracer, memory_seed=None):
    """Build a scripted agent that calls calculator then gives a final answer."""
    client = FakeLLMClient([
        LLMResponse(
            text="",
            tool_calls=(
                ToolCall(id="tc_1", name="calculator", arguments={"expression": "12 * 9"}),
            ),
        ),
        LLMResponse(text="The result of 12 * 9 is 108."),
    ])
    memory = InMemoryLongTerm()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RecallTool(memory))
    return ReActAgent(
        llm=client,
        tools=registry,
        system_prompt="You are a helpful assistant.",
        tracer=tracer,
    )


def _make_direct_agent(tracer, memory_seed=None):
    """Build a scripted agent that answers without calling any tool."""
    client = FakeLLMClient([LLMResponse(text="Hello there!")])
    memory = InMemoryLongTerm()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RecallTool(memory))
    return ReActAgent(
        llm=client,
        tools=registry,
        system_prompt="You are a helpful assistant.",
        tracer=tracer,
    )


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
