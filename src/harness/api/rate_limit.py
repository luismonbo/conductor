"""Rate-limit configuration: a single slowapi Limiter plus the two dynamic
tier callables used as @limiter.limit(...) arguments in main.py.

Limit values are resolved fresh on every request via a callable rather than
a literal string baked in at import time, so HARNESS_RATE_LIMIT_* changes
(including the ones test fixtures set after this module is first imported)
take effect immediately. The identity key (get_remote_address, i.e. the raw
socket IP -- there's no auth yet) is isolated here so switching to an
API-key-based key once auth ships is a one-line change, not a rebuild. See
docs/superpowers/specs/2026-08-27-rate-limiting-design.md.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from harness.config.settings import get_settings

# Effectively unlimited: used instead of a real toggle so "disabled" reuses
# the same dynamic-callable mechanism as the real tier limits, rather than
# depending on a second, less-certain slowapi feature (dynamically
# re-checked enable/disable).
_DISABLED_LIMIT = "1000000/minute"

# No storage_uri: defaults to slowapi/limits' in-memory backend, matching
# the current single-process deployment (no Redis in this stack).
limiter = Limiter(key_func=get_remote_address)


def strict_limit() -> str:
    settings = get_settings()
    return settings.rate_limit_strict if settings.rate_limit_enabled else _DISABLED_LIMIT


def default_limit() -> str:
    settings = get_settings()
    return settings.rate_limit_default if settings.rate_limit_enabled else _DISABLED_LIMIT
