"""remember/recall are opt-in: a public deployment must not expose shared memory."""
from harness.agents.default.tools import build_registry


def _tool_names(memory_tools: bool) -> set[str]:
    registry = build_registry(memory_tools=memory_tools)
    return {spec.name for spec in registry.specs()}


def test_memory_tools_absent_by_default():
    names = _tool_names(memory_tools=False)
    assert "remember" not in names
    assert "recall" not in names


def test_calculator_survives_with_memory_disabled():
    assert "calculator" in _tool_names(memory_tools=False)


def test_memory_tools_present_when_enabled():
    names = _tool_names(memory_tools=True)
    assert "remember" in names
    assert "recall" in names
