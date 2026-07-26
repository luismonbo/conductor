"""RagEvalCase and RagDataset — a sibling to evaluation/harness/dataset.py's
EvalCase, not an extension of it. The two 'expected' shapes (tool call vs.
relevant chunks/reference answer) don't overlap."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RagExpected:
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_document_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""


@dataclass(frozen=True)
class RagEvalCase:
    id: str
    query: str
    expected: RagExpected
    tags: list[str] = field(default_factory=list)


class RagDataset:
    def __init__(self, cases: list[RagEvalCase], version: str = "1.0") -> None:
        self.cases = cases
        self.version = version

    def filter_by_tags(self, tags: list[str]) -> "RagDataset":
        if not tags:
            return self
        matched = [c for c in self.cases if any(t in c.tags for t in tags)]
        return RagDataset(matched, self.version)

    @classmethod
    def load(cls, path: Path) -> "RagDataset":
        raw = json.loads(path.read_text())
        cases: list[RagEvalCase] = []
        for item in raw["cases"]:
            exp_raw = item.get("expected", {})
            expected = RagExpected(
                relevant_chunk_ids=exp_raw.get("relevant_chunk_ids", []),
                relevant_document_ids=exp_raw.get("relevant_document_ids", []),
                reference_answer=exp_raw.get("reference_answer", ""),
            )
            cases.append(
                RagEvalCase(
                    id=item["id"],
                    query=item["query"],
                    expected=expected,
                    tags=item.get("tags", []),
                )
            )
        return cls(cases, version=raw.get("version", "1.0"))