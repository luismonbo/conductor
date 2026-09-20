"""An opaque sid cookie identifies an anonymous visitor across requests."""
import httpx2

import harness.api.main as _main
from harness.api.session import COOKIE_NAME


async def test_first_response_sets_httponly_cookie(client_with_fake):
    client, _ = client_with_fake
    response = await client.get("/health")

    set_cookie = response.headers.get("set-cookie", "")
    assert f"{COOKIE_NAME}=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie.replace("samesite", "SameSite")


async def test_existing_cookie_is_reused_not_reminted(client_with_fake):
    client, _ = client_with_fake
    first = await client.get("/health")
    sid = first.cookies.get(COOKIE_NAME)
    assert sid

    second = await client.get("/health")
    # Cookie already present on the request, so nothing new is issued.
    assert second.headers.get("set-cookie") is None
    assert client.cookies.get(COOKIE_NAME) == sid


async def test_separate_clients_get_separate_ids():
    transport = httpx2.ASGITransport(app=_main.app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as a:
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as b:
            await a.get("/health")
            await b.get("/health")
            assert a.cookies.get(COOKIE_NAME) != b.cookies.get(COOKIE_NAME)
