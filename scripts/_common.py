"""What db.py and rag.py share: one-line errors, safe text, tables and JSON, and
Postgres connections that never print a password.

Imported as `_common`: running a script puts scripts/ first on sys.path, and
pytest adds it through the pythonpath in pyproject.toml.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row
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


# --- Postgres ------------------------------------------------------------------

_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}
# Where a connection string keeps its password: URL userinfo (up to the last "@",
# so an unencoded "@" or "/" stays inside it) or a password= keyword or parameter.
_URL_USERINFO = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<userinfo>[^?#]*)@")
_PASSWORD_PARAM = re.compile(
    r"(?P<key>password\s*=\s*)(?P<value>'(?:[^'\\]|\\.)*'|[^\s&]+)", re.IGNORECASE
)


def first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text.splitlines()[0]}" if text else type(exc).__name__


def _url_password(dsn: str) -> str | None:
    match = _URL_USERINFO.match(dsn)
    if match is None or ":" not in match.group("userinfo"):
        return None
    return match.group("userinfo").split(":", 1)[1]


def _without_password(dsn: str) -> str:
    """The DSN with its password replaced by ***, however malformed it is."""
    match = _URL_USERINFO.match(dsn)
    if match is not None and _url_password(dsn) is not None:
        user = match.group("userinfo").split(":", 1)[0]
        dsn = f"{match.group('scheme')}{user}:***@{dsn[match.end():]}"
    return _PASSWORD_PARAM.sub(lambda m: m.group("key") + "***", dsn)


def _scrub(text: str, dsn: str) -> str:
    """`text` without the DSN's password or any piece of it libpq split off."""
    secrets = [m.group("value").strip("'") for m in _PASSWORD_PARAM.finditer(dsn)]
    if (password := _url_password(dsn)) is not None:
        secrets.append(password)
    pieces = {
        piece
        for secret in secrets
        for piece in (secret, *re.split(r"[@/:?#%&]", secret))
        if len(piece) >= 3
    }
    for piece in sorted(pieces, key=len, reverse=True):
        text = text.replace(piece, "***")
    return text


def postgres_server(dsn: str) -> str:
    """user@host:port/db, never the password."""
    try:
        info = conninfo_to_dict(_without_password(dsn))
    except psycopg.Error:
        return "the configured server"
    host = info.get("host") or "localhost"
    port = info.get("port") or 5432
    return f"{info.get('user', '?')}@{host}:{port}/{info.get('dbname', '?')}"


def _is_local(dsn: str) -> bool:
    try:
        return (conninfo_to_dict(_without_password(dsn)).get("host") or "") in _LOCAL_HOSTS
    except psycopg.Error:
        return False


def connect_postgres(dsn: str) -> psycopg.Connection[DictRow]:
    """An autocommit connection with dict rows, or a one-line ScriptError that names
    the server and never the password."""
    malformed = (
        f"malformed connection string for {postgres_server(dsn)}: URL-encode special "
        "characters in the password (@ is %40, / is %2F, % is %25)"
    )
    # libpq would read an unencoded "@" or "/" as the host, and look the rest of
    # the password up in DNS: refuse before connecting.
    if re.search(r"[@/?#]", _url_password(dsn) or ""):
        raise ScriptError(malformed)
    try:
        return psycopg.connect(
            dsn, autocommit=True, connect_timeout=10, row_factory=dict_row
        )
    except psycopg.ProgrammingError as exc:  # libpq can't parse it
        raise ScriptError(malformed) from exc
    except psycopg.Error as exc:
        hint = (
            "start it with `make infra`"
            if _is_local(dsn)
            else "check the connection string, and that the server accepts this machine"
        )
        reason = _scrub(first_line(exc), dsn)
        raise ScriptError(
            f"can't reach Postgres at {postgres_server(dsn)}: {reason} ({hint})"
        ) from exc
