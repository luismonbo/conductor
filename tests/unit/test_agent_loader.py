"""load_agent: import an agent's definition by folder name, one clear error per failure."""
from __future__ import annotations

import pytest

from harness.agents.default.prompt import SYSTEM_PROMPT
from harness.agents.definition import AgentDefinition
from harness.orchestration.agents import AgentLoadError, load_agent

FIXTURES = "tests.fixtures.agents"


def test_loads_a_committed_agent_by_name():
    definition = load_agent("default")
    assert isinstance(definition, AgentDefinition)
    assert definition.prompt == SYSTEM_PROMPT


def test_loads_an_agent_from_another_package():
    definition = load_agent("good", package=FIXTURES)
    assert definition.prompt == "You are the good fixture agent."
    assert definition.model == "fixture-model"


def test_an_unknown_name_says_where_it_looked():
    with pytest.raises(AgentLoadError, match=r"no agent named 'nope'.*nope/agent\.py"):
        load_agent("nope", package=FIXTURES)


@pytest.mark.parametrize(
    "name", ["a.b", "../x", "Website", "website ", "", "9lives", "web-site"]
)
def test_a_name_that_is_not_a_plain_identifier_is_rejected(name):
    with pytest.raises(AgentLoadError, match="invalid"):
        load_agent(name, package=FIXTURES)


def test_the_invalid_name_error_shows_the_exact_value():
    # A trailing space from a hand-edited .env must be visible in the message.
    with pytest.raises(AgentLoadError, match="'website '"):
        load_agent("website ")


def test_a_module_that_is_not_an_agent_folder_is_not_an_agent():
    # harness/agents/definition.py is a module, not an agent folder.
    with pytest.raises(AgentLoadError, match="no agent named 'definition'"):
        load_agent("definition")


def test_a_folder_without_agent_py_is_not_an_agent():
    with pytest.raises(AgentLoadError, match="no agent named 'no_agent_module'"):
        load_agent("no_agent_module", package=FIXTURES)


def test_agent_py_must_define_agent():
    with pytest.raises(AgentLoadError, match="must define AGENT"):
        load_agent("no_agent_attr", package=FIXTURES)


def test_agent_must_be_an_agent_definition():
    with pytest.raises(AgentLoadError, match="must define AGENT"):
        load_agent("wrong_type", package=FIXTURES)


def test_the_definition_name_must_match_the_folder():
    with pytest.raises(AgentLoadError, match="must match its folder"):
        load_agent("mismatched", package=FIXTURES)


def test_an_import_error_inside_an_agent_is_not_reported_as_missing():
    with pytest.raises(ModuleNotFoundError) as excinfo:
        load_agent("broken", package=FIXTURES)
    assert excinfo.value.name == "harness.does_not_exist"
