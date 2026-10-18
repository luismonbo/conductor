"""configure_logging: the API's own INFO lines reach stderr, and no library can log a
visitor's message, session id or IP address, whatever the root logger's level."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.observability.logging_setup import configure_logging

_SRC = Path(__file__).resolve().parents[2] / "src"

_BARE_PROCESS = """
import logging
from harness.observability.logging_setup import configure_logging

configure_logging("INFO")
configure_logging("INFO")
logging.getLogger("harness.probe").info("harness info line")
logging.getLogger("harness.probe").debug("harness debug line")
logging.getLogger("somelib").info("library info line")
logging.getLogger("somelib").warning("library warning line")
"""


@pytest.fixture(scope="module")
def bare_stderr() -> str:
    """stderr of a fresh interpreter whose root logger nothing has configured,
    which is what the app sees under plain uvicorn."""
    # coloredlogs' .pth file configures root at interpreter start when this is set.
    env = {key: value for key, value in os.environ.items() if key != "COLOREDLOGS_AUTO_INSTALL"}
    result = subprocess.run(
        [sys.executable, "-c", _BARE_PROCESS],
        capture_output=True,
        text=True,
        timeout=30,
        env={**env, "PYTHONPATH": str(_SRC)},
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def _lines_with(text: str, output: str) -> list[str]:
    return [line for line in output.splitlines() if text in line]


def test_harness_info_lines_reach_stderr_once(bare_stderr):
    [line] = _lines_with("harness info line", bare_stderr)
    assert "INFO" in line
    assert "harness.probe" in line
    assert "harness debug line" not in bare_stderr


def test_other_libraries_stay_at_warning(bare_stderr):
    assert "library info line" not in bare_stderr
    [line] = _lines_with("library warning line", bare_stderr)
    assert "somelib" in line


def test_an_already_configured_root_logger_is_left_alone(caplog):
    caplog.set_level(logging.ERROR)  # an operator's setup: handlers and a level on root
    root = logging.getLogger()
    handlers = list(root.handlers)

    configure_logging("INFO")

    assert root.handlers == handlers
    assert root.level == logging.ERROR


# Where each SDK writes request bodies or SQL parameters at DEBUG.
_SDK_BODY_LOGGERS = ["typesafe_sdk", "openai._base_client", "aiosqlite", "langfuse"]


@pytest.mark.parametrize("sdk_logger", _SDK_BODY_LOGGERS)
def test_a_root_debug_config_cannot_make_an_sdk_log_a_visitor_message(sdk_logger, caplog):
    caplog.set_level(logging.DEBUG)  # a root-level DEBUG config
    configure_logging("DEBUG")

    logging.getLogger(sdk_logger).debug("request body: %s", "a visitor's message")

    assert "a visitor's message" not in caplog.text


@pytest.mark.parametrize("sdk_logger", _SDK_BODY_LOGGERS)
def test_pinned_sdks_still_log_their_warnings(sdk_logger, caplog):
    configure_logging("INFO")

    logging.getLogger(sdk_logger).warning("an SDK warning")

    assert "an SDK warning" in caplog.text


def test_slowapi_warnings_are_silenced_but_its_errors_are_not(caplog):
    """slowapi's 429 warning names the limiter key, which is the visitor's IP."""
    configure_logging("INFO")

    logging.getLogger("slowapi").warning(
        "ratelimit 15 per 1 minute (203.0.113.7) exceeded at endpoint: /chat/stream"
    )
    logging.getLogger("slowapi").error("Skipping limit: x")

    assert "203.0.113.7" not in caplog.text
    assert "Skipping limit" in caplog.text
