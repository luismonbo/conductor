"""EvalCase dataclass and Dataset loader."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExpectedToolCall:
    name: str


@dataclass(frozen=True)
class Expected:
    tool_call: ExpectedToolCall | None = None
    # Key presence only — not values, because LLM arg phrasing varies.
    tool_args: dict[str, object] | None = None
    # AND-semantics: all strings must appear in the final output.
    output_contains: list[str] = field(default_factory=list)
    # OR-semantics: at least one string must appear in the final output.
    output_contains_any: list[str] = field(default_factory=list)
    # When True, asserts that the agent called no tools at all.
    no_tool_call: bool = False


@dataclass(frozen=True)
class EvalCase:
    id: str
    description: str
    input: str
    tags: list[str]
    expected: Expected
    memory_seed: list[str] = field(default_factory=list)


class Dataset:
    def __init__(self, cases: list[EvalCase], version: str = "1.0") -> None:
        self.cases = cases
        self.version = version

    def filter_by_tags(self, tags: list[str]) -> "Dataset":
        if not tags:
            return self
        matched = [c for c in self.cases if any(t in c.tags for t in tags)]
        return Dataset(matched, self.version)

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        raw = json.loads(path.read_text())
        cases: list[EvalCase] = []
        for item in raw["cases"]:
            exp_raw = item.get("expected", {})
            tc_raw = exp_raw.get("tool_call")
            ta_raw = exp_raw.get("tool_args")
            expected = Expected(
                tool_call=ExpectedToolCall(name=tc_raw["name"]) if tc_raw else None,
                tool_args=ta_raw,
                output_contains=exp_raw.get("output_contains", []),
                output_contains_any=exp_raw.get("output_contains_any", []),
                no_tool_call=bool(exp_raw.get("no_tool_call", False)),
            )
            cases.append(
                EvalCase(
                    id=item["id"],
                    description=item.get("description", ""),
                    input=item["input"],
                    tags=item.get("tags", []),
                    expected=expected,
                    memory_seed=item.get("memory_seed", []),
                )
            )
        return cls(cases, version=raw.get("version", "1.0"))
