from __future__ import annotations

import json
from pathlib import Path

from evaluation.rag.dataset import RagDataset

_DATASET_PATH = Path(__file__).parent.parent.parent / "evaluation" / "rag" / "datasets" / "papers_v1.json"


def test_shipped_dataset_loads_and_is_well_formed():
    dataset = RagDataset.load(_DATASET_PATH)

    assert dataset.version == "1.0"
    assert dataset.cases, "papers_v1.json should carry real cases"
    assert len({c.id for c in dataset.cases}) == len(dataset.cases), "case ids must be unique"
    for case in dataset.cases:
        assert case.query.strip(), f"{case.id} has an empty query"
        assert case.tags, f"{case.id} has no tags (needed for --tags filtering)"


def test_labelled_chunk_ids_belong_to_their_declared_document():
    """chunk_id is '{document_id}:{index}'. A label pointing at a chunk from a
    different document is a copy-paste slip that would silently report a
    retrieval failure that never happened."""
    dataset = RagDataset.load(_DATASET_PATH)

    for case in dataset.cases:
        for chunk_id in case.expected.relevant_chunk_ids:
            document_id = chunk_id.rsplit(":", 1)[0]
            assert document_id in case.expected.relevant_document_ids, (
                f"{case.id}: chunk {chunk_id} is not covered by "
                f"relevant_document_ids {case.expected.relevant_document_ids}"
            )


def test_dataset_filters_by_tag():
    dataset = RagDataset.load(_DATASET_PATH)

    smoke = dataset.filter_by_tags(["smoke"])

    assert smoke.cases
    assert len(smoke.cases) < len(dataset.cases), "smoke should be a strict subset"


def test_an_empty_dataset_still_loads_cleanly(tmp_path):
    """The CLI reports 'no cases to run' and exits 1 rather than crashing —
    the Phase 0 acceptance path, still reachable now the shipped set is full."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"version": "1.0", "cases": []}))

    dataset = RagDataset.load(path)

    assert dataset.cases == []
    assert dataset.filter_by_tags(["anything"]).cases == []
