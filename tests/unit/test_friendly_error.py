"""Connection and provider failures get a human-readable message."""
from harness.api.main import _friendly_error


def test_connection_errors_get_gateway_hint():
    class APIConnectionError(Exception):  # matched by class NAME, not import
        pass

    msg = _friendly_error(APIConnectionError("[Errno 61] Connection refused"))
    assert "LLM gateway" in msg
    assert "make up" in msg


def test_other_errors_pass_through():
    assert _friendly_error(ValueError("boom")) == "boom"


def test_rate_limit_errors_get_generic_unavailable_message():
    class RateLimitError(Exception):  # matched by class NAME, not import
        pass

    msg = _friendly_error(RateLimitError("429 quota exceeded, deployment=prod-gpt5"))
    assert msg == "The assistant is temporarily unavailable. Please try again later."


def test_permission_denied_errors_get_generic_unavailable_message():
    class PermissionDeniedError(Exception):
        pass

    msg = _friendly_error(PermissionDeniedError("403 access disabled"))
    assert msg == "The assistant is temporarily unavailable. Please try again later."


def test_authentication_errors_get_generic_unavailable_message():
    class AuthenticationError(Exception):
        pass

    msg = _friendly_error(AuthenticationError("401 invalid api key"))
    assert msg == "The assistant is temporarily unavailable. Please try again later."
