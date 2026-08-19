"""Unit tests for RAG-related factory functions in orchestration/build.py."""
from __future__ import annotations

from pathlib import Path

import yaml

from harness.orchestration.build import list_collections


def test_list_collections_empty_directory_returns_empty_list(tmp_path: Path):
    assert list_collections(tmp_path) == []


def test_list_collections_reads_collection_field_from_each_manifest(tmp_path: Path):
    (tmp_path / "papers.yaml").write_text(yaml.safe_dump({"collection": "papers"}))
    (tmp_path / "manuals.yaml").write_text(yaml.safe_dump({"collection": "manuals"}))

    assert list_collections(tmp_path) == ["manuals", "papers"]


def test_list_collections_missing_directory_returns_empty_list(tmp_path: Path):
    assert list_collections(tmp_path / "does_not_exist") == []
