from __future__ import annotations

from pathlib import Path

from evaluation.rag.dataset import RagDataset

_DATASET_PATH = Path(__file__).parent.parent.parent / "evaluation" / "rag" / "datasets" / "papers_v1.json"


def test_starter_dataset_is_valid_and_currently_empty():
    dataset = RagDataset.load(_DATASET_PATH)

    assert dataset.version == "1.0"
    assert dataset.cases == []
