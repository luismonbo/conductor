"""Every proxy that injects the shared API key must also own X-Visitor-IP.

The API trusts X-Visitor-IP on any request that carries the valid key, on the
understanding that such a request came through the proxy that set the header.
A proxy that injects the key but passes the header through would let a caller
pick their own rate-limit bucket. The Cloudflare Worker (site repo) sets the
header; the dev-server proxies (this repo's frontend and the site's) strip it.
docs/superpowers/specs/2026-10-18-production-proxy-design.md (section 5)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_frontend_dev_proxy_strips_the_visitor_ip_header():
    vite_config = (ROOT / "frontend" / "vite.config.ts").read_text()

    assert "removeHeader('x-visitor-ip')" in vite_config
