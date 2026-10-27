"""The sid cookie's lifetime: a year by default, the browser session when asked."""
from starlette.responses import Response

from harness.api.session import COOKIE_NAME, attach


def _set_cookie(**kwargs) -> str:
    response = Response()
    attach(response, "abc123", secure=True, **kwargs)
    return response.headers["set-cookie"]


def _attributes(set_cookie: str) -> set[str]:
    """The cookie's attributes, lowercased, without its name=value pair."""
    return {part.strip().lower() for part in set_cookie.split(";")[1:]}


def test_the_cookie_lasts_a_year_by_default():
    assert "max-age=31536000" in _attributes(_set_cookie())


def test_a_session_cookie_has_neither_max_age_nor_expires():
    attributes = _attributes(_set_cookie(persistent=False))

    assert not any(a.startswith(("max-age", "expires")) for a in attributes)


def test_a_session_cookie_keeps_every_other_protection():
    set_cookie = _set_cookie(persistent=False)

    assert set_cookie.startswith(f"{COOKIE_NAME}=abc123")
    assert {"httponly", "secure", "samesite=lax", "path=/"} <= _attributes(set_cookie)
