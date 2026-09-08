"""Load + validate a single deployment target's Settings overrides."""
from __future__ import annotations

import pytest

from harness.config.targets import SECRET_FIELDS, resolve_target_overrides


def test_loads_overrides_from_a_target_file(tmp_path):
    (tmp_path / "acme.yaml").write_text("llm_backend: azure\nrag_collection: acme-docs\n")

    overrides = resolve_target_overrides("acme", base_dir=tmp_path)

    assert overrides == {"llm_backend": "azure", "rag_collection": "acme-docs"}


def test_raises_when_target_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="acme"):
        resolve_target_overrides("acme", base_dir=tmp_path)


def test_raises_when_target_sets_a_secret_field(tmp_path):
    (tmp_path / "acme.yaml").write_text("azure_api_key: shh\n")

    with pytest.raises(ValueError, match="azure_api_key"):
        resolve_target_overrides("acme", base_dir=tmp_path)


def test_raises_when_target_file_is_not_a_mapping(tmp_path):
    (tmp_path / "acme.yaml").write_text("- just\n- a\n- list\n")

    with pytest.raises(ValueError, match="mapping"):
        resolve_target_overrides("acme", base_dir=tmp_path)


def test_empty_target_file_yields_no_overrides(tmp_path):
    (tmp_path / "acme.yaml").write_text("")

    assert resolve_target_overrides("acme", base_dir=tmp_path) == {}


def test_secret_fields_is_exactly_the_expected_set():
    # Regression lock: adding/removing a credential-shaped Settings field
    # must be a deliberate edit here, not a silent gap.
    assert SECRET_FIELDS == {
        "llm_api_key",
        "azure_api_key",
        "embedding_api_key",
        "memory_url",
        "pgvector_url",
        "checkpointer_url",
    }


@pytest.mark.parametrize("field", sorted(SECRET_FIELDS))
def test_every_secret_field_is_individually_rejected(tmp_path, field):
    (tmp_path / "acme.yaml").write_text(f"{field}: shh\n")

    with pytest.raises(ValueError):
        resolve_target_overrides("acme", base_dir=tmp_path)
