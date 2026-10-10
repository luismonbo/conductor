"""Fixture: an agent whose own code imports a module that does not exist."""
import harness.does_not_exist  # noqa: F401
