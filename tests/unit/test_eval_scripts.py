"""The eval-dataset scripts still find the datasets after moving into scripts/eval/."""
from __future__ import annotations

from pathlib import Path

import scripts.eval.label_assist as label_assist
import scripts.eval.refresh_corpus_fingerprint as refresh_corpus_fingerprint

_DATASETS = Path(__file__).resolve().parents[2] / "evaluation" / "rag" / "datasets"


def test_label_assist_writes_into_the_datasets_folder():
    assert _DATASETS.is_dir()
    assert label_assist._DATASETS_DIR == _DATASETS


def test_refresh_reads_from_the_datasets_folder():
    assert refresh_corpus_fingerprint._DATASETS_DIR == _DATASETS
