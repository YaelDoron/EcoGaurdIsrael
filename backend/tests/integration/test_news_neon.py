"""Live integration test for wildfire news persistence against Neon."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from src.config.settings import settings
from src.database.connection import get_engine, get_session, init_db
from src.models.fire_report import WildfireReport
from src.repositories.news_repository import NewsRepository

pytestmark = pytest.mark.integration

TEST_SOURCE_URL = "https://example.com/ecoguard-integration-test-wildfire-report"
TEST_SOURCE_FEED = "ECOGUARD_INTEGRATION_TEST_FEED"
TEST_PUBLISHED_AT = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
TEST_FETCHED_AT = datetime(2026, 9, 12, 10, 5, tzinfo=timezone.utc)


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM wildfire_reports "
                "WHERE source_url = :source_url OR source_feed = :source_feed"
            ),
            {"source_url": TEST_SOURCE_URL, "source_feed": TEST_SOURCE_FEED},
        )


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url) -> None:
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def make_report(**overrides) -> WildfireReport:
    defaults = dict(
        source_url=TEST_SOURCE_URL,
        source_feed=TEST_SOURCE_FEED,
        title="ECOGUARD_INTEGRATION_TEST_TITLE",
        summary="Temporary wildfire news integration test row.",
        location_name="Haifa",
        latitude=32.794,
        longitude=34.9896,
        published_at=TEST_PUBLISHED_AT,
        fetched_at=TEST_FETCHED_AT,
    )
    defaults.update(overrides)
    return WildfireReport(**defaults)


def _count_test_rows() -> int:
    with get_session() as session:
        return session.execute(
            text(
                "SELECT COUNT(*) FROM wildfire_reports "
                "WHERE source_url = :source_url OR source_feed = :source_feed"
            ),
            {"source_url": TEST_SOURCE_URL, "source_feed": TEST_SOURCE_FEED},
        ).scalar_one()


def test_news_table_save_read_duplicate_and_cleanup_against_neon() -> None:
    init_db()
    repository = NewsRepository()

    with get_session() as session:
        table_exists = session.execute(
            text(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'wildfire_reports'"
                ")"
            )
        ).scalar_one()
    assert table_exists is True

    first = repository.save_report(make_report())
    assert first.is_duplicate is False

    stored = repository.get_by_source_url(TEST_SOURCE_URL)
    assert stored is not None
    assert stored.source_url == TEST_SOURCE_URL
    assert stored.source_feed == TEST_SOURCE_FEED
    assert stored.location_name == "Haifa"
    assert stored.published_at == TEST_PUBLISHED_AT
    assert stored.fetched_at == TEST_FETCHED_AT

    second = repository.save_report(make_report(summary="changed"))
    assert second.is_duplicate is True
    assert second.report.summary == "Temporary wildfire news integration test row."
    assert _count_test_rows() == 1

    _delete_test_rows()
    assert _count_test_rows() == 0
