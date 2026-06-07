"""Unit tests for Dataset.load() and tag filtering."""
import json
import tempfile
from pathlib import Path

import pytest

from evaluation.harness.dataset import Dataset


_MINIMAL_JSON = {
    "version": "1.0",
    "cases": [
        {
            "id": "calc_001",
            "description": "Calculator test",
            "tags": ["calculator", "arithmetic"],
            "input": "what is 12 * 9?",
            "expected": {
                "tool_call": {"name": "calculator"},
                "tool_args": {"expression": None},
                "output_contains": ["108"],
            },
        },
        {
            "id": "direct_001",
            "description": "Direct answer",
            "tags": ["direct"],
            "input": "say hello",
            "expected": {
                "output_contains": ["hello"],
            },
        },
    ],
}


def _write_dataset(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


class TestDatasetLoad:
    def test_loads_version(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        assert ds.version == "1.0"

    def test_loads_all_cases(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        assert len(ds.cases) == 2

    def test_parses_tool_call(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        calc = ds.cases[0]
        assert calc.expected.tool_call is not None
        assert calc.expected.tool_call.name == "calculator"

    def test_parses_tool_args(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        calc = ds.cases[0]
        assert calc.expected.tool_args == {"expression": None}

    def test_parses_output_contains(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        calc = ds.cases[0]
        assert calc.expected.output_contains == ["108"]

    def test_absent_tool_call_is_none(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        direct = ds.cases[1]
        assert direct.expected.tool_call is None

    def test_absent_tool_args_is_none(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        direct = ds.cases[1]
        assert direct.expected.tool_args is None

    def test_parses_tags(self):
        path = _write_dataset(_MINIMAL_JSON)
        ds = Dataset.load(path)
        assert "calculator" in ds.cases[0].tags

    def test_ignores_unknown_fields(self):
        data = dict(_MINIMAL_JSON)
        data["cases"] = [dict(data["cases"][0], unknown_field="ignored")]
        path = _write_dataset(data)
        ds = Dataset.load(path)
        assert len(ds.cases) == 1


class TestDatasetFilterByTags:
    def _load(self) -> Dataset:
        return Dataset.load(_write_dataset(_MINIMAL_JSON))

    def test_empty_tags_returns_all(self):
        ds = self._load().filter_by_tags([])
        assert len(ds.cases) == 2

    def test_filter_by_single_tag(self):
        ds = self._load().filter_by_tags(["calculator"])
        assert len(ds.cases) == 1
        assert ds.cases[0].id == "calc_001"

    def test_filter_by_tag_with_no_match_returns_empty(self):
        ds = self._load().filter_by_tags(["nonexistent"])
        assert len(ds.cases) == 0

    def test_filter_matches_any_tag(self):
        ds = self._load().filter_by_tags(["direct", "calculator"])
        assert len(ds.cases) == 2
