"""Logging for the API process, configured once at startup (api/main.py's lifespan).

uvicorn configures only its own loggers. Without this, the harness's INFO lines
(the input guard's one line per blocked message among them) were dropped, and
every other line reached stderr without a level or a logger name.
"""
from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Libraries whose DEBUG lines carry a visitor's message or session id: TypeSafe,
# OpenAI and Langfuse log request bodies, aiosqlite every SQL parameter. Held at
# WARNING whatever the root logger's level (Langfuse already holds its own there;
# listed in case a release stops). Their own switches (TYPESAFE_LOG_LEVEL,
# OPENAI_LOG, LANGFUSE_DEBUG) take effect when the SDK loads, after startup has
# run this, so they still override it: never set them in production.
_PRIVATE_AT_DEBUG = ("typesafe_sdk", "openai", "aiosqlite", "langfuse")


def configure_logging(level: str) -> None:
    """Send harness.* lines at `level`, and every other library's at WARNING, to stderr.

    A root logger something else already configured (uvicorn's --log-config,
    pytest) keeps its handlers and level.
    """
    logging.basicConfig(format=_FORMAT, level=logging.WARNING)
    logging.getLogger("harness").setLevel(level)
    for name in _PRIVATE_AT_DEBUG:
        logging.getLogger(name).setLevel(logging.WARNING)
