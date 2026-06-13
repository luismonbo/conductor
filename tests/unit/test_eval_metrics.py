"""Unit tests for the three deterministic eval metrics.

No agent, no network — metrics receive pre-built AgentRunResult objects.
"""
import pytest

from harness.observability.tracer import TraceCollector

from evaluation.harness.dataset import EvalCase, Expected, ExpectedToolCall
from evaluation.harness.runner import AgentRunResult
from evaluation.metrics.arg_schema import ArgSchemaMetric
from evaluation.metrics.no_tool_call import NoToolCallMetric
from evaluation.metrics.output_contains import OutputContainsMetric
from evaluation.metrics.output_contains_any import OutputContainsAnyMetric
from evaluation.metrics.tool_call import ToolCallMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case(
    id: str = "test_001",
    input: str = "test input",
    tool_call: str | None = None,
    tool_args: dict | None = None,
    output_contains: list[str] | None = None,
    output_contains_any: list[str] | None = None,
    no_tool_call: bool = False,
) -> EvalCase:
    return EvalCase(
        id=id,
        description="",
        input=input,
        tags=[],
        expected=Expected(
            tool_call=ExpectedToolCall(name=tool_call) if tool_call else None,
            tool_args=tool_args,
            output_contains=output_contains or [],
            output_contains_any=output_contains_any or [],
            no_tool_call=no_tool_call,
        ),
    )


def _result(
    output: str = "",
    tool_names: list[str] | None = None,
    tool_args: dict[str, dict] | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        output=output,
        tool_names_called=tool_names or [],
        tool_args_by_name=tool_args or {},
    )


_tracer = TraceCollector()


# ---------------------------------------------------------------------------
# ToolCallMetric
# ---------------------------------------------------------------------------

class TestToolCallMetric:
    metric = ToolCallMetric()

    def test_passes_when_no_expectation(self):
        result = self.metric.score(_case(), _result(), _tracer)
        assert result.passed
        assert "skipped" in result.reason

    def test_passes_when_expected_tool_was_called(self):
        case = _case(tool_call="calculator")
        result = self.metric.score(case, _result(tool_names=["calculator"]), _tracer)
        assert result.passed
        assert result.score == 1.0

    def test_fails_when_no_tools_called(self):
        case = _case(tool_call="calculator")
        result = self.metric.score(case, _result(tool_names=[]), _tracer)
        assert not result.passed
        assert result.score == 0.0
        assert "no tools were called" in result.reason

    def test_fails_when_wrong_tool_called(self):
        case = _case(tool_call="calculator")
        result = self.metric.score(case, _result(tool_names=["recall"]), _tracer)
        assert not result.passed
        assert "calculator" in result.reason

    def test_passes_when_expected_tool_among_multiple_calls(self):
        case = _case(tool_call="calculator")
        result = self.metric.score(
            case, _result(tool_names=["recall", "calculator"]), _tracer
        )
        assert result.passed


# ---------------------------------------------------------------------------
# ArgSchemaMetric
# ---------------------------------------------------------------------------

class TestArgSchemaMetric:
    metric = ArgSchemaMetric()

    def test_skips_when_no_tool_args_expectation(self):
        case = _case(tool_call="calculator")
        result = self.metric.score(case, _result(), _tracer)
        assert result.passed
        assert "skipped" in result.reason

    def test_skips_when_no_tool_call_expectation(self):
        result = self.metric.score(_case(), _result(), _tracer)
        assert result.passed

    def test_passes_when_all_expected_keys_present(self):
        case = _case(tool_call="calculator", tool_args={"expression": None})
        res = _result(
            tool_names=["calculator"],
            tool_args={"calculator": {"expression": "12 * 9"}},
        )
        result = self.metric.score(case, res, _tracer)
        assert result.passed
        assert result.score == 1.0

    def test_fails_when_expected_key_missing(self):
        case = _case(tool_call="calculator", tool_args={"expression": None})
        res = _result(
            tool_names=["calculator"],
            tool_args={"calculator": {"wrong_key": "value"}},
        )
        result = self.metric.score(case, res, _tracer)
        assert not result.passed
        assert "expression" in result.reason

    def test_fails_when_tool_was_not_called(self):
        case = _case(tool_call="calculator", tool_args={"expression": None})
        result = self.metric.score(case, _result(tool_names=[]), _tracer)
        assert not result.passed
        assert "not called" in result.reason

    def test_passes_with_extra_keys_in_actual_args(self):
        case = _case(tool_call="calculator", tool_args={"expression": None})
        res = _result(
            tool_names=["calculator"],
            tool_args={"calculator": {"expression": "1+1", "extra": "ignored"}},
        )
        result = self.metric.score(case, res, _tracer)
        assert result.passed


# ---------------------------------------------------------------------------
# OutputContainsMetric
# ---------------------------------------------------------------------------

class TestOutputContainsMetric:
    metric = OutputContainsMetric()

    def test_skips_when_no_expectation(self):
        result = self.metric.score(_case(), _result(), _tracer)
        assert result.passed
        assert "skipped" in result.reason

    def test_passes_when_all_substrings_present(self):
        case = _case(output_contains=["108"])
        result = self.metric.score(case, _result(output="The answer is 108."), _tracer)
        assert result.passed
        assert result.score == 1.0

    def test_passes_case_insensitive(self):
        case = _case(output_contains=["hello"])
        result = self.metric.score(case, _result(output="Hello there!"), _tracer)
        assert result.passed

    def test_fails_when_substring_missing(self):
        case = _case(output_contains=["108"])
        result = self.metric.score(case, _result(output="I don't know."), _tracer)
        assert not result.passed
        assert "108" in result.reason

    def test_and_semantics_all_must_match(self):
        case = _case(output_contains=["108", "multiplication"])
        result = self.metric.score(
            case, _result(output="The answer is 108."), _tracer
        )
        assert not result.passed
        assert "multiplication" in result.reason

    def test_passes_when_all_multiple_substrings_present(self):
        case = _case(output_contains=["108", "multiplication"])
        result = self.metric.score(
            case, _result(output="multiplication gives 108"), _tracer
        )
        assert result.passed


# ---------------------------------------------------------------------------
# NoToolCallMetric
# ---------------------------------------------------------------------------

class TestNoToolCallMetric:
    metric = NoToolCallMetric()

    def test_skips_when_no_expectation(self):
        result = self.metric.score(_case(), _result(), _tracer)
        assert result.passed
        assert "skipped" in result.reason

    def test_passes_when_no_tools_called(self):
        case = _case(no_tool_call=True)
        result = self.metric.score(case, _result(tool_names=[]), _tracer)
        assert result.passed
        assert result.score == 1.0

    def test_fails_when_a_tool_was_called(self):
        case = _case(no_tool_call=True)
        result = self.metric.score(case, _result(tool_names=["calculator"]), _tracer)
        assert not result.passed
        assert result.score == 0.0
        assert "calculator" in result.reason

    def test_fails_listing_all_called_tools(self):
        case = _case(no_tool_call=True)
        result = self.metric.score(
            case, _result(tool_names=["recall", "calculator"]), _tracer
        )
        assert not result.passed
        assert "recall" in result.reason
        assert "calculator" in result.reason

    def test_skips_when_assertion_is_false(self):
        case = _case(no_tool_call=False)
        result = self.metric.score(case, _result(tool_names=["calculator"]), _tracer)
        assert result.passed
        assert "skipped" in result.reason


# ---------------------------------------------------------------------------
# OutputContainsAnyMetric
# ---------------------------------------------------------------------------

class TestOutputContainsAnyMetric:
    metric = OutputContainsAnyMetric()

    def test_skips_when_no_expectation(self):
        result = self.metric.score(_case(), _result(), _tracer)
        assert result.passed
        assert "skipped" in result.reason

    def test_passes_when_first_alternative_matches(self):
        case = _case(output_contains_any=["7", "seven"])
        result = self.metric.score(case, _result(output="There are 7 continents."), _tracer)
        assert result.passed
        assert result.score == 1.0

    def test_passes_when_second_alternative_matches(self):
        case = _case(output_contains_any=["7", "seven"])
        result = self.metric.score(case, _result(output="There are seven continents."), _tracer)
        assert result.passed
        assert result.score == 1.0

    def test_passes_case_insensitive(self):
        case = _case(output_contains_any=["Au"])
        result = self.metric.score(case, _result(output="The symbol is au."), _tracer)
        assert result.passed

    def test_fails_when_no_alternative_matches(self):
        case = _case(output_contains_any=["7", "seven"])
        result = self.metric.score(case, _result(output="There are many continents."), _tracer)
        assert not result.passed
        assert result.score == 0.0
        assert "7" in result.reason and "seven" in result.reason
