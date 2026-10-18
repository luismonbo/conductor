"""The shared-key check that require_api_key and the rate limiter's key function both use.

docs/superpowers/specs/2026-10-18-production-proxy-design.md (section 7)
"""
from __future__ import annotations

import pytest
from starlette.requests import Request

from harness.api.auth import has_valid_api_key


def _request(authorization: bytes | None) -> Request:
    headers = [] if authorization is None else [(b"authorization", authorization)]
    return Request({"type": "http", "headers": headers})


@pytest.mark.parametrize(
    "authorization, api_key, expected",
    [
        (b"Bearer k1", "k1", True),
        (b"Bearer wrong", "k1", False),
        (b"Bearer ", "k1", False),
        (b"Basic k1", "k1", False),
        (b"k1", "k1", False),
        (None, "k1", False),
        (b"Bearer caf\xe9", "k1", False),  # a non-ASCII token must not raise
        (b"Bearer k1", "", False),  # an empty configured key never matches
    ],
)
def test_has_valid_api_key(authorization, api_key, expected):
    assert has_valid_api_key(_request(authorization), api_key) is expected
