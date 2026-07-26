from __future__ import annotations

import json

from evaluation.rag.dataset import RagDataset, RagEvalCase, RagExpected


def test_load_parses_cases_from_json(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({
        "version": "1.0",
        "cases": [
            {
                "id": "q1",
                "query": "what method is used?",
                "tags": ["smoke"],
                "expected": {
                    "relevant_chunk_ids": ["doc1:0"],
                    "reference_answer": "self-attention",
                },
            }
        ],
    }))

    dataset = RagDataset.load(path)

    assert len(dataset.cases) == 1
    assert dataset.cases[0].query == "what method is used?"
    assert dataset.cases[0].expected.relevant_chunk_ids == ["doc1:0"]
    assert dataset.cases[0].expected.reference_answer == "self-attention"
    assert dataset.cases[0].tags == ["smoke"]


def test_load_defaults_expected_fields_when_absent(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"version": "1.0", "cases": [{"id": "q1", "query": "anything?"}]}))

    dataset = RagDataset.load(path)

    assert dataset.cases[0].expected.relevant_chunk_ids == []
    assert dataset.cases[0].expected.relevant_document_ids == []
    assert dataset.cases[0].expected.reference_answer == ""
    assert dataset.cases[0].tags == []


def test_filter_by_tags_matches_any():
    cases = [
        RagEvalCase(id="a", query="q1", expected=RagExpected(), tags=["smoke"]),
        RagEvalCase(id="b", query="q2", expected=RagExpected(), tags=["deep"]),
    ]
    dataset = RagDataset(cases)

    filtered = dataset.filter_by_tags(["smoke"])

    assert [c.id for c in filtered.cases] == ["a"]


def test_filter_by_tags_empty_list_returns_all():
    cases = [RagEvalCase(id="a", query="q1", expected=RagExpected(), tags=["smoke"])]
    dataset = RagDataset(cases)

    assert dataset.filter_by_tags([]).cases == cases
