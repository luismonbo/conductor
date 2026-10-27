"""An opaque sid cookie identifies an anonymous visitor across requests."""
import httpx2
from starlette.requests import Request

import harness.api.main as _main
from harness.api.session import COOKIE_NAME, read_or_mint


def _attributes(set_cookie: str) -> set[str]:
    """The cookie's attributes, without its name=value pair."""
    return {part.strip() for part in set_cookie.split(";")[1:]}


async def test_first_response_sets_httponly_cookie(client_with_fake):
    client, _ = client_with_fake
    response = await client.get("/health")

    set_cookie = response.headers.get("set-cookie", "")
    assert f"{COOKIE_NAME}=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie.replace("samesite", "SameSite")
    assert "Max-Age=31536000" in set_cookie
    assert "Path=/" in set_cookie


async def test_cookie_is_secure_by_default(client_with_fake, monkeypatch, tmp_path):
    # Assert the shipped default, not a local opt-out: drop any value
    # load_dotenv copied into the process, and step away from the .env file
    # Settings would otherwise read directly.
    monkeypatch.delenv("HARNESS_SESSION_COOKIE_SECURE", raising=False)
    monkeypatch.chdir(tmp_path)
    client, _ = client_with_fake

    set_cookie = (await client.get("/health")).headers["set-cookie"]

    assert "Secure" in _attributes(set_cookie)


async def test_secure_flag_can_be_disabled_for_plain_http_development(
    client_with_fake, monkeypatch
):
    monkeypatch.setenv("HARNESS_SESSION_COOKIE_SECURE", "false")
    client, _ = client_with_fake

    set_cookie = (await client.get("/health")).headers["set-cookie"]

    assert set_cookie.startswith(f"{COOKIE_NAME}=")
    assert "Secure" not in _attributes(set_cookie)


async def test_cookie_is_a_session_cookie_when_persistence_is_off(client_with_fake, monkeypatch):
    # The public site's target turns this off, so the browser drops sid when it closes.
    monkeypatch.setenv("HARNESS_SESSION_COOKIE_PERSISTENT", "false")
    client, _ = client_with_fake

    set_cookie = (await client.get("/health")).headers["set-cookie"]

    assert set_cookie.startswith(f"{COOKIE_NAME}=")
    assert not any(a.startswith(("Max-Age", "Expires")) for a in _attributes(set_cookie))


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
    async with httpx2.AsyncClient(transport=transport, base_url="https://test") as a:
        async with httpx2.AsyncClient(transport=transport, base_url="https://test") as b:
            await a.get("/health")
            await b.get("/health")
            assert a.cookies.get(COOKIE_NAME) != b.cookies.get(COOKIE_NAME)


def test_minted_session_id_carries_256_bits():
    # The id is deliberately unsigned; that ruling rests on it being
    # unguessable. secrets.token_urlsafe(32) is 43 base64url characters.
    sid, minted = read_or_mint(Request({"type": "http", "headers": []}))
    assert minted
    assert len(sid) >= 43
