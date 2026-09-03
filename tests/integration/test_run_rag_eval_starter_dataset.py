from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.rag.dataset import RagDataset

_DATASET_PATH = (
    Path(__file__).parent.parent.parent
    / "evaluation"
    / "rag"
    / "datasets"
    / "papers_v2.json"
)

_SKIP_REASON = (
    "papers_v2.json does not exist yet — it lands in Task 11 (Build the "
    "dataset — HUMAN IN THE LOOP). This suite turns real automatically once "
    "that file is created."
)


@pytest.fixture
def v2_dataset() -> RagDataset:
    if not _DATASET_PATH.exists():
        pytest.skip(_SKIP_REASON)
    return RagDataset.load(_DATASET_PATH)


def test_shipped_dataset_loads_and_is_well_formed(v2_dataset):
    assert v2_dataset.version == "2.0"
    assert v2_dataset.cases, "papers_v2.json should carry real cases"
    ids = {c.id for c in v2_dataset.cases}
    assert len(ids) == len(v2_dataset.cases), "case ids must be unique"
    for case in v2_dataset.cases:
        assert case.query.strip(), f"{case.id} has an empty query"
    assert v2_dataset.corpus.collection, "corpus fingerprint needs a collection"
    assert v2_dataset.corpus.documents, "corpus fingerprint needs verified documents"


def test_labelled_chunk_ids_belong_to_their_declared_document(v2_dataset):
    """chunk_id is '{document_id}:{index}'. A label pointing at a chunk from a
    different document is a copy-paste slip that would silently report a
    retrieval failure that never happened.

    Only grade>=2 chunks are checked (via the `relevant_chunk_ids` property,
    which already applies that threshold) — grade-1 chunks are deliberately
    exempt because the labelling tool only records a document in
    `relevant_document_ids` once it has a grade>=2 chunk.
    """
    for case in v2_dataset.cases:
        for chunk_id in case.expected.relevant_chunk_ids:
            document_id = chunk_id.rsplit(":", 1)[0]
            assert document_id in case.expected.relevant_document_ids, (
                f"{case.id}: chunk {chunk_id} is not covered by "
                f"relevant_document_ids {case.expected.relevant_document_ids}"
            )


def test_dataset_filters_by_suite(v2_dataset):
    smoke = v2_dataset.filter_by_suite("smoke")

    assert smoke.cases
    assert len(smoke.cases) < len(v2_dataset.cases), "smoke should be a strict subset"


def test_an_empty_dataset_still_loads_cleanly(tmp_path):
    """The CLI reports 'no cases to run' and exits 1 rather than crashing —
    the Phase 0 acceptance path, still reachable now the shipped set is full."""
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "corpus": {"collection": "papers", "documents": {}, "verified_at": ""},
                "cases": [],
            }
        )
    )

    dataset = RagDataset.load(path)

    assert dataset.cases == []
    assert dataset.filter_by_suite("anything").cases == []
