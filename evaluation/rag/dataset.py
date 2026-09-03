"""RagEvalCase and RagDataset — schema v2.

v2 replaces v1's binary `relevant_chunk_ids` list with a graded 0-3 map, so
nDCG and the binary metrics read one source of truth rather than two lists
that can drift apart. It also carries provenance, an orthogonal taxonomy,
and a corpus fingerprint recording what the labels were verified against.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# A chunk counts as "relevant" for the binary metrics (recall, MRR) at grade
# 2 or above. Grades 3 and 2 contribute to the answer; grade 1 is on-topic
# context that does not. See datasets/RUBRIC.md.
RELEVANT_GRADE_THRESHOLD = 2


@dataclass(frozen=True)
class RagExpected:
    graded_chunks: dict[str, int] = field(default_factory=dict)
    relevant_document_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""

    @property
    def relevant_chunk_ids(self) -> list[str]:
        return [
            cid
            for cid, grade in self.graded_chunks.items()
            if grade >= RELEVANT_GRADE_THRESHOLD
        ]


@dataclass(frozen=True)
class RagProvenance:
    source: str = "synthetic"  # synthetic | production | human
    verified_by: str = ""
    verified_at: str = ""


@dataclass(frozen=True)
class RagCorpus:
    """What the labels were verified against. Compared to the live index at
    run time so a re-ingest that invalidates labels fails loudly instead of
    silently reporting false failures."""

    collection: str = ""
    # document_id -> chunk count
    documents: dict[str, int] = field(default_factory=dict)
    verified_at: str = ""


@dataclass(frozen=True)
class RagEvalCase:
    id: str
    query: str
    expected: RagExpected
    provenance: RagProvenance = field(default_factory=RagProvenance)
    query_type: str = ""
    difficulty: str = ""
    expected_behaviour: str = ""
    suites: list[str] = field(default_factory=list)
    excluded_from: list[str] = field(default_factory=list)


class RagDataset:
    def __init__(
        self,
        cases: list[RagEvalCase],
        version: str = "2.0",
        corpus: RagCorpus | None = None,
    ) -> None:
        self.cases = cases
        self.version = version
        self.corpus = corpus or RagCorpus()

    def __iter__(self) -> Iterator[RagEvalCase]:
        return iter(self.cases)

    def filter_by_suite(self, suite: str | None) -> "RagDataset":
        if not suite:
            return self
        matched = [c for c in self.cases if suite in c.suites]
        return RagDataset(matched, self.version, self.corpus)

    def exclude_gate(self, gate: str | None) -> "RagDataset":
        if not gate:
            return self
        matched = [c for c in self.cases if gate not in c.excluded_from]
        return RagDataset(matched, self.version, self.corpus)

    @classmethod
    def load(cls, path: Path) -> "RagDataset":
        raw = json.loads(path.read_text())
        version = str(raw.get("version", "1.0"))
        if not version.startswith("2"):
            raise ValueError(
                f"{path.name} is schema v{version}; the harness requires schema v2. "
                "v1 labels were invalidated by the SP1 re-ingest — rebuild the dataset "
                "rather than migrating it."
            )

        corpus_raw = raw.get("corpus", {})
        corpus = RagCorpus(
            collection=corpus_raw.get("collection", ""),
            documents={k: int(v) for k, v in corpus_raw.get("documents", {}).items()},
            verified_at=corpus_raw.get("verified_at", ""),
        )

        cases: list[RagEvalCase] = []
        for item in raw["cases"]:
            exp_raw = item.get("expected", {})
            if "relevant_chunk_ids" in exp_raw:
                raise ValueError(
                    f"case {item['id']}: 'relevant_chunk_ids' is schema v1. "
                    "Use 'graded_chunks': {chunk_id: 0-3}."
                )
            prov_raw = item.get("provenance", {})
            cases.append(
                RagEvalCase(
                    id=item["id"],
                    query=item["query"],
                    expected=RagExpected(
                        graded_chunks={
                            k: int(v) for k, v in exp_raw.get("graded_chunks", {}).items()
                        },
                        relevant_document_ids=exp_raw.get("relevant_document_ids", []),
                        reference_answer=exp_raw.get("reference_answer", ""),
                    ),
                    provenance=RagProvenance(
                        source=prov_raw.get("source", "synthetic"),
                        verified_by=prov_raw.get("verified_by", ""),
                        verified_at=prov_raw.get("verified_at", ""),
                    ),
                    query_type=item.get("query_type", ""),
                    difficulty=item.get("difficulty", ""),
                    expected_behaviour=item.get("expected_behaviour", ""),
                    suites=item.get("suites", []),
                    excluded_from=item.get("excluded_from", []),
                )
            )
        return cls(cases, version=version, corpus=corpus)
