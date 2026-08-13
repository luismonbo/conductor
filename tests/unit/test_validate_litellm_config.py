"""litellm_config.yaml sanity: parseable, non-empty, unique model names."""
from pathlib import Path

from scripts.validate_litellm_config import validate_config


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return str(p)


def test_valid_config_passes(tmp_path):
    path = _write(tmp_path, """
model_list:
  - model_name: local-gemma
    litellm_params:
      model: ollama/gemma3
""")
    assert validate_config(path) == []


def test_duplicate_names_rejected(tmp_path):
    path = _write(tmp_path, """
model_list:
  - model_name: claude
    litellm_params: {model: anthropic/claude-sonnet-4-5}
  - model_name: claude
    litellm_params: {model: anthropic/claude-haiku-4-5}
""")
    assert any("duplicate" in p for p in validate_config(path))


def test_empty_model_list_rejected(tmp_path):
    path = _write(tmp_path, "model_list: []\n")
    assert any("empty" in p for p in validate_config(path))


def test_missing_litellm_model_rejected(tmp_path):
    path = _write(tmp_path, """
model_list:
  - model_name: broken
    litellm_params: {}
""")
    assert any("broken" in p for p in validate_config(path))


def test_repo_config_is_valid():
    repo_config = Path(__file__).parents[2] / "litellm_config.yaml"
    assert validate_config(str(repo_config)) == []
