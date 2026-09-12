"""Unit tests for NewsRepository using SQLite in-memory."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.database.models.wildfire_report_db import WildfireReportDB
from src.models.fire_report import WildfireReport
from src.repositories.exceptions import NewsRepositoryError
from src.repositories.news_repository import NewsRepository, SaveNewsReportResult

PUBLISHED_AT = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
FETCHED_AT = datetime(2026, 9, 12, 10, 5, tzinfo=timezone.utc)


@pytest.fixture
def repository(sqlite_session_factory) -> NewsRepository:
    return NewsRepository(session_factory=sqlite_session_factory)


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


def count_reports(sqlite_session_factory) -> int:
    session = sqlite_session_factory()
    count = len(session.execute(select(WildfireReportDB)).scalars().all())
    session.close()
    return count


def test_new_report_saves_successfully(repository):
    result = repository.save_report(make_report())

    assert isinstance(result, SaveNewsReportResult)
    assert result.is_duplicate is False
    assert result.report.title == "Wildfire reported near Haifa"
    assert result.report.published_at == PUBLISHED_AT
    assert result.report.fetched_at == FETCHED_AT


def test_exists_by_source_url(repository):
    report = make_report()
    repository.save_report(report)

    assert repository.exists_by_source_url(report.source_url) is True
    assert repository.exists_by_source_url("https://example.com/missing") is False


def test_same_source_url_twice_creates_one_row_and_duplicate_result(repository, sqlite_session_factory):
    first = repository.save_report(make_report(summary="original"))
    second = repository.save_report(make_report(summary="changed"))

    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert second.report.summary == "original"
    assert count_reports(sqlite_session_factory) == 1


def test_get_by_source_url_round_trips_domain_values(repository):
    report = make_report(
        source_feed=None,
        summary=None,
        location_name=None,
        latitude=None,
        longitude=None,
        published_at=None,
    )
    repository.save_report(report)

    stored = repository.get_by_source_url(report.source_url)

    assert stored == report


def test_published_at_none_round_trips(repository):
    report = make_report(published_at=None)
    repository.save_report(report)

    stored = repository.get_by_source_url(report.source_url)

    assert stored is not None
    assert stored.published_at is None
    assert stored.fetched_at == FETCHED_AT


def test_legacy_postgres_string_timestamp_is_mapped_to_aware_datetime():
    parsed = NewsRepository._ensure_aware_datetime("2026-09-12 10:00:00+00")

    assert parsed == PUBLISHED_AT


def test_missing_report_returns_none(repository):
    assert repository.get_by_source_url("https://example.com/missing") is None


def test_integrity_error_race_fallback_returns_duplicate_result(repository, monkeypatch, sqlite_session_factory):
    repository.save_report(make_report(summary="original"))

    original_find = NewsRepository._find_by_source_url
    call_count = {"n": 0}

    def fake_find_by_source_url(session, source_url):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None
        return original_find(session, source_url)

    monkeypatch.setattr(
        NewsRepository,
        "_find_by_source_url",
        staticmethod(fake_find_by_source_url),
    )

    result = repository.save_report(make_report(summary="changed"))

    assert result.is_duplicate is True
    assert result.report.summary == "original"
    assert count_reports(sqlite_session_factory) == 1


def test_unrecoverable_integrity_error_is_converted(repository, monkeypatch):
    repository.save_report(make_report())

    def fake_find_by_source_url(session, source_url):
        return None

    monkeypatch.setattr(
        NewsRepository,
        "_find_by_source_url",
        staticmethod(fake_find_by_source_url),
    )

    with pytest.raises(NewsRepositoryError):
        repository.save_report(make_report())


def test_repository_session_is_usable_after_rolled_back_write(repository, sqlite_session_factory):
    raw_session = sqlite_session_factory()
    raw_session.add(
        WildfireReportDB(
            source_url=None,
            title="Invalid",
            fetched_at=FETCHED_AT,
        )
    )
    with pytest.raises(IntegrityError):
        raw_session.commit()
    raw_session.rollback()
    raw_session.close()

    result = repository.save_report(make_report())

    assert result.is_duplicate is False
