"""The daily token budget counts from UTC midnight."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.api.main import _utc_midnight


def test_utc_midnight_is_the_start_of_todays_utc_day():
    midnight = _utc_midnight()
    assert midnight.utcoffset() == timedelta(0)
    assert (midnight.hour, midnight.minute, midnight.second, midnight.microsecond) == (0, 0, 0, 0)
    assert timedelta(0) <= datetime.now(timezone.utc) - midnight < timedelta(days=1)
