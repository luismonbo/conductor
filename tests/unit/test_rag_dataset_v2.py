from __future__ import annotations

import json

import pytest

from evaluation.rag.dataset import RagDataset

_CASE = {
    "id": "attn_encoder_layers",
    "query": "How many layers are in the encoder stack?",
    "expected": {
        "graded_chunks": {"papers/50455611c3ed8b48:5": 3, "papers/50455611c3ed8b48:6": 1},
        "relevant_document_ids": ["papers/50455611c3ed8b48"],
        "reference_answer": "A stack of N = 6 identical layers.",
    },
    "provenance": {"source": "human", "verified_by": "luis", "verified_at": "2026-09-01"},
    "query_type": "factual",
    "difficulty": "easy",
    "expected_behaviour": "grounded_answer",
    "suites": ["smoke"],
    "excluded_from": [],
}

_DOC = {
    "version": "2.0",
    "corpus": {
        "collection": "papers",
        "documents": {"papers/50455611c3ed8b48": 43},
        "verified_at": "2026-09-01",
    },
    "cases": [_CASE],
}


def _write(tmp_path, doc):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(doc))
    return path


def test_loads_graded_chunks_and_taxonomy(tmp_path):
    ds = RagDataset.load(_write(tmp_path, _DOC))

    case = ds.cases[0]
    assert case.expected.graded_chunks["papers/50455611c3ed8b48:5"] == 3
    assert case.query_type == "factual"
    assert case.provenance.verified_by == "luis"
    assert ds.corpus.documents == {"papers/50455611c3ed8b48": 43}


def test_relevant_chunk_ids_applies_grade_two_threshold(tmp_path):
    ds = RagDataset.load(_write(tmp_path, _DOC))

    # grade 3 qualifies, grade 1 does not
    assert ds.cases[0].expected.relevant_chunk_ids == ["papers/50455611c3ed8b48:5"]


def test_rejects_v1_schema_with_clear_message(tmp_path):
    v1 = {
        "version": "1.0",
        "cases": [{"id": "x", "query": "q", "expected": {"relevant_chunk_ids": ["a:1"]}}],
    }

    with pytest.raises(ValueError, match="schema v2"):
        RagDataset.load(_write(tmp_path, v1))


def test_rejects_case_level_v1_field_even_under_v2_version(tmp_path):
    # Top-level version says 2.0, but this one case still carries the v1
    # field — e.g. a case copy-pasted from the old dataset and never migrated.
    mixed = dict(
        _DOC,
        cases=[{"id": "x", "query": "q", "expected": {"relevant_chunk_ids": ["a:1"]}}],
    )

    with pytest.raises(ValueError, match="schema v1"):
        RagDataset.load(_write(tmp_path, mixed))


def test_filter_by_suite_and_exclude_gate(tmp_path):
    other = dict(_CASE, id="other", suites=[], excluded_from=["sp2"])
    ds = RagDataset.load(_write(tmp_path, dict(_DOC, cases=[_CASE, other])))

    assert [c.id for c in ds.filter_by_suite("smoke")] == ["attn_encoder_layers"]
    assert [c.id for c in ds.exclude_gate("sp2")] == ["attn_encoder_layers"]
    assert [c.id for c in ds.filter_by_suite(None)] == ["attn_encoder_layers", "other"]
