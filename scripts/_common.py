"""What db.py and rag.py share: one-line errors, safe text, tables and JSON.

Imported as `_common`: running a script puts scripts/ first on sys.path, and
pytest adds it through the pythonpath in pyproject.toml.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from harness.config.settings import Settings

# Stored text can be untrusted (a visitor's message, the sid cookie a visitor
# sends): raw ESC, BEL, CR, C1 controls and bidi overrides would rewrite the
# terminal that prints them (erase rows, retitle the window, spoof OSC 8 links).
# Text output shows them as escapes; newlines and tabs stay.
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
# json.dumps already escapes C0 controls; these are the ones it leaves raw.
_JSON_CONTROLS = re.compile(r"[\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


class ScriptError(Exception):
    """A failure reported as one line on stderr, with exit code 1."""


def visible(text: str) -> str:
    """`text` with every terminal control shown as an escape instead of acted on."""

    def escape(match: re.Match[str]) -> str:
        code = ord(match.group())
        return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"

    return _CONTROLS.sub(escape, text)


def cut(text: str, width: int) -> str:
    """One line of at most `width` characters; newlines become spaces."""
    line = " ".join(visible(text).split())
    return line if len(line) <= width else line[: width - 1] + "…"


def print_json(data: Any) -> None:
    """JSON with the exact text, but no raw control a terminal would act on."""
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    print(_JSON_CONTROLS.sub(lambda match: f"\\u{ord(match.group()):04x}", text))


def print_table(
    headers: list[str] | None,
    rows: list[list[str]],
    numeric: frozenset[int] = frozenset(),
    indent: str = "",
) -> None:
    """Aligned columns; `numeric` columns align right."""
    lines = [headers, *rows] if headers else rows
    if not lines:
        return
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    for line in lines:
        cells = [
            cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i])
            for i, cell in enumerate(line)
        ]
        print(indent + "  ".join(cells).rstrip())


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def resolve(prefix: str, candidates: list[str], what: str) -> str:
    """The one candidate `prefix` names: an exact id, or a unique prefix of one."""
    if prefix in candidates:
        return prefix
    matches = sorted({c for c in candidates if c.startswith(prefix)})
    if not matches:
        raise ScriptError(f"no {what} matching {prefix!r}")
    if len(matches) > 1:
        listed = "\n  ".join(visible(match) for match in matches[:10])
        raise ScriptError(f"{prefix!r} matches {len(matches)} {what}s:\n  {listed}")
    return matches[0]


def load_settings(read: Callable[[], Settings]) -> Settings:
    """The settings `read` returns, or a one-line ScriptError saying why not."""
    try:
        return read()
    except ValidationError as exc:
        # One line, and never the input: it holds every HARNESS_* value, secrets too.
        reasons = "; ".join(
            error["msg"] for error in exc.errors(include_url=False, include_input=False)
        )
        raise ScriptError(f"settings: {reasons}") from exc
    except ValueError as exc:
        raise ScriptError(f"settings: {exc}") from exc
