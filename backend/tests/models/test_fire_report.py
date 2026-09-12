"""Unit tests for the WildfireReport model."""
from datetime import datetime, timezone

import pytest

from src.models.fire_report import WildfireReport

PUBLISHED_AT = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
FETCHED_AT = datetime(2026, 9, 12, 10, 5, tzinfo=timezone.utc)


def make_report(**overrides) -> WildfireReport:
    defaults = dict(
        source_url="https://example.com/news/wildfire-1",
        source_feed="Example Feed",
        title="Wildfire reported near Haifa",
        summary="Firefighters responded to a forest fire.",
        location_name="Haifa",
        latitude=32.794,
        longitude=34.9896,
        published_at=PUBLISHED_AT,
        fetched_at=FETCHED_AT,
    )
    defaults.update(overrides)
    return WildfireReport(**defaults)


def test_valid_timezone_aware_datetimes_are_accepted():
    report = make_report()

    assert report.published_at == PUBLISHED_AT
    assert report.fetched_at == FETCHED_AT


def test_published_at_none_is_allowed():
    report = make_report(published_at=None)

    assert report.published_at is None


def test_published_at_must_be_datetime_or_none():
    with pytest.raises(ValueError):
        make_report(published_at="Sat, 12 Sep 2026 10:00:00 GMT")


def test_fetched_at_is_required_datetime():
    with pytest.raises(ValueError):
        make_report(fetched_at="2026-09-12T10:05:00+00:00")
