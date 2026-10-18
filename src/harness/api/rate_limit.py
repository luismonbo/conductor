"""Rate-limit configuration: a single slowapi Limiter plus the two dynamic
tier callables used as @limiter.limit(...) arguments in main.py.

Limit values are resolved fresh on every request via a callable rather than
a literal string baked in at import time, so HARNESS_RATE_LIMIT_* changes
(including the ones test fixtures set after this module is first imported)
take effect immediately.

The identity key is visitor_key: the socket address, except for a request that
carries the valid shared API key and an X-Visitor-IP header. Only the proxy in
front of the API holds the key (the Cloudflare Worker in production, Caddy and
the dev-server proxies locally), so "carries the valid key" means "came through
that proxy", and the header it sets names the real visitor. Without this every
visitor on Azure Container Apps shares one bucket, because the socket address
there is the platform's ingress proxy. A proxy that injects the key must set or
strip the header. See docs/superpowers/specs/2026-08-27-rate-limiting-design.md
and docs/superpowers/specs/2026-10-18-production-proxy-design.md.
"""
from __future__ import annotations

import ipaddress

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from harness.api.auth import has_valid_api_key
from harness.config.settings import get_settings

# The header the proxy sets to the visitor's IP (lower-case, as Starlette names headers).
VISITOR_IP_HEADER = "x-visitor-ip"

# An ISP hands an IPv6 customer a /64 or more, so keying on single addresses
# would let one customer rotate through 2**64 buckets. Cloudflare counts IPv6
# per /64 for the same reason.
_IPV6_BUCKET_PREFIX = 64

# Effectively unlimited: used instead of a real toggle so "disabled" reuses
# the same dynamic-callable mechanism as the real tier limits, rather than
# depending on a second, less-certain slowapi feature (dynamically
# re-checked enable/disable).
_DISABLED_LIMIT = "1000000/minute"

def normalize_visitor_ip(value: str | None) -> str | None:
    """The limiter key for an X-Visitor-IP value, or None when it is unusable.

    An IPv4 address is its own key, an IPv4-mapped IPv6 address becomes the
    IPv4 address, and any other IPv6 address becomes its /64. Anything that is
    not exactly one plain IP address (empty, junk, a comma list, leading zeros,
    a scoped address such as fe80::1%eth0) is None, and the caller falls back
    to the socket address.
    """
    if value is None:
        return None
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv4Address):
        return str(address)
    if address.scope_id is not None:
        # Python reduces a scoped address without complaint, but a zone id
        # never comes from Cloudflare.
        return None
    if address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(ipaddress.ip_network((address, _IPV6_BUCKET_PREFIX), strict=False))


def visitor_key(request: Request) -> str:
    """The identity a request is rate-limited under.

    slowapi passes the request to a key function only if the parameter is
    literally named ``request``, and it silently skips the limit for an empty
    key: this never returns "".
    """
    settings = get_settings()
    if settings.auth_enabled and has_valid_api_key(request, settings.api_key):
        visitor = normalize_visitor_ip(request.headers.get(VISITOR_IP_HEADER))
        if visitor is not None:
            return visitor
    return get_remote_address(request)


# No storage_uri: defaults to slowapi/limits' in-memory backend, matching
# the current single-process deployment (no Redis in this stack).
limiter = Limiter(key_func=visitor_key)


def strict_limit() -> str:
    settings = get_settings()
    return settings.rate_limit_strict if settings.rate_limit_enabled else _DISABLED_LIMIT


def default_limit() -> str:
    settings = get_settings()
    return settings.rate_limit_default if settings.rate_limit_enabled else _DISABLED_LIMIT
