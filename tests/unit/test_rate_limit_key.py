"""The rate limiter's key: the visitor IP a key-injecting proxy vouches for, else the socket address.

docs/superpowers/specs/2026-10-18-production-proxy-design.md (section 7)
"""
from __future__ import annotations

import inspect

import pytest
from starlette.requests import Request

from harness.api.rate_limit import VISITOR_IP_HEADER, normalize_visitor_ip, visitor_key

SOCKET = "10.0.0.9"
VISITOR = "203.0.113.7"
BEARER = {"Authorization": "Bearer test-key"}


@pytest.fixture(autouse=True)
def _auth_on(monkeypatch):
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "true")
    monkeypatch.setenv("HARNESS_API_KEY", "test-key")


def _request(
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = (SOCKET, 5555),
) -> Request:
    scope = {
        "type": "http",
        # Starlette hands headers to the app as lower-cased latin-1 bytes.
        "headers": [(k.lower().encode(), v.encode("latin-1")) for k, v in (headers or {}).items()],
        "client": client,
    }
    return Request(scope)


def test_the_header_name_is_the_one_the_worker_sets():
    assert VISITOR_IP_HEADER == "x-visitor-ip"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("203.0.113.7", "203.0.113.7"),
        ("  203.0.113.7 ", "203.0.113.7"),
        ("2001:db8:1234:5678:aaaa:bbbb:cccc:dddd", "2001:db8:1234:5678::/64"),
        ("2001:db8:1234:5678:1::1", "2001:db8:1234:5678::/64"),  # same /64, same key
        ("::ffff:203.0.113.7", "203.0.113.7"),  # a mapped address becomes the IPv4
        (None, None),
        ("", None),
        ("   ", None),
        ("not-an-ip", None),
        ("203.0.113.7, 198.51.100.2", None),  # two values
        ("001.002.003.004", None),  # leading zeros
        ("fe80::1%eth0", None),  # scoped: Python reduces it without error, so reject it explicitly
        ("٢٠٣.0.113.7", None),  # non-ASCII digits
        ("A" * 4096, None),
    ],
)
def test_normalize_visitor_ip(value, expected):
    assert normalize_visitor_ip(value) == expected


def test_a_keyed_request_is_limited_by_the_visitor_ip():
    assert visitor_key(_request({**BEARER, "X-Visitor-IP": VISITOR})) == VISITOR


def test_visitors_in_different_ipv6_64s_get_different_keys():
    first = visitor_key(_request({**BEARER, "X-Visitor-IP": "2001:db8:1234:5678::1"}))
    second = visitor_key(_request({**BEARER, "X-Visitor-IP": "2001:db8:1234:5679::1"}))

    assert first != second


@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong", "Bearer ", "Basic test-key"],
    ids=["no-key", "wrong-key", "empty-token", "wrong-scheme"],
)
def test_the_header_is_ignored_without_the_valid_key(authorization):
    headers = {"X-Visitor-IP": VISITOR}
    if authorization is not None:
        headers["Authorization"] = authorization

    assert visitor_key(_request(headers)) == SOCKET


def test_the_header_is_ignored_when_auth_is_off(monkeypatch):
    monkeypatch.setenv("HARNESS_AUTH_ENABLED", "false")

    assert visitor_key(_request({**BEARER, "X-Visitor-IP": VISITOR})) == SOCKET


@pytest.mark.parametrize("header", [None, "garbage"], ids=["absent", "garbage"])
def test_a_keyed_request_without_a_usable_header_falls_back_to_the_socket(header):
    headers = dict(BEARER)
    if header is not None:
        headers["X-Visitor-IP"] = header

    assert visitor_key(_request(headers)) == SOCKET


@pytest.mark.parametrize(
    "request_",
    [
        _request(client=None),
        _request(client=("", 0)),
        _request({**BEARER, "X-Visitor-IP": "A" * 4096}, client=None),
        _request({**BEARER, "X-Visitor-IP": "1.2.3.4, 5.6.7.8"}, client=None),
        _request({**BEARER, "X-Visitor-IP": "fe80::1%eth0"}, client=None),
    ],
    ids=["no-client", "empty-host", "junk-header", "comma-list", "scoped-ipv6"],
)
def test_the_key_is_never_empty(request_):
    # slowapi silently skips the limit for an empty key.
    assert visitor_key(request_) != ""


def test_a_request_with_no_client_is_keyed_on_localhost():
    assert visitor_key(_request(client=None)) == "127.0.0.1"


def test_the_key_function_takes_a_parameter_named_request():
    # slowapi passes the request to a key function only if the parameter is literally named `request`.
    assert list(inspect.signature(visitor_key).parameters) == ["request"]
