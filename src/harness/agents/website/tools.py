"""Tools for the website's public assistant: read-only access to its wiki."""
from __future__ import annotations

from harness.adapters.tools.wiki import ReadWikiIndexTool, ReadWikiPageTool
from harness.agents.definition import ToolDeps
from harness.core.tools.registry import ToolRegistry


def build_tools(deps: ToolDeps) -> ToolRegistry:
    """The wiki is required: answering from it is the website assistant's whole
    job, so it does not start without it."""
    if deps.wiki is None:
        raise RuntimeError(
            "the website agent needs its wiki: set HARNESS_WIKI_DIR, or "
            "HARNESS_WIKI_BLOB_ACCOUNT_URL and HARNESS_WIKI_BLOB_CONTAINER"
        ) from deps.wiki_error
    registry = ToolRegistry()
    registry.register(ReadWikiIndexTool(deps.wiki))
    registry.register(ReadWikiPageTool(deps.wiki))
    return registry
