"""Connection failures to the LLM gateway get a human-readable message."""
from harness.api.main import _friendly_error


def test_connection_errors_get_gateway_hint():
    class APIConnectionError(Exception):  # matched by class NAME, not import
        pass

    msg = _friendly_error(APIConnectionError("[Errno 61] Connection refused"))
    assert "LLM gateway" in msg
    assert "make up" in msg


def test_other_errors_pass_through():
    assert _friendly_error(ValueError("boom")) == "boom"
