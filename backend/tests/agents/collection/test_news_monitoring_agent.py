"""Unit tests for NewsMonitoringAgent with mocked dependencies."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

from src.agents.collection.news_monitoring_agent import PROJECT_ROOT, NewsMonitoringAgent
from src.models.fire_report import WildfireReport
from src.repositories.exceptions import NewsRepositoryError
from src.repositories.news_repository import NewsRepository, SaveNewsReportResult

EXPECTED_PUBLISHED_AT = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)


def make_entry(**overrides) -> dict:
    defaults = {
        "source_feed": "Example Feed",
        "title": "Wildfire near Haifa",
        "summary": "Firefighters responded to a forest fire.",
        "link": "https://example.com/news/wildfire-1",
        "published": "Sat, 12 Sep 2026 10:00:00 GMT",
    }
    defaults.update(overrides)
    return defaults


def make_agent(
    rss_fetcher: Mock | None = None,
    text_processor: Mock | None = None,
    geocoder: Mock | None = None,
    news_repository: Mock | None = None,
) -> NewsMonitoringAgent:
    rss_fetcher = rss_fetcher or Mock()
    text_processor = text_processor or Mock()
    geocoder = geocoder or Mock()
    news_repository = news_repository or Mock(spec=NewsRepository)

    return NewsMonitoringAgent(
        rss_fetcher=rss_fetcher,
        text_processor=text_processor,
        geocoder=geocoder,
        news_repository=news_repository,
        enable_file_logging=False,
    )


def _wire_success(text_processor: Mock, geocoder: Mock, news_repository: Mock) -> None:
    text_processor.is_relevant.return_value = True
    text_processor.extract_location.return_value = "Haifa"
    geocoder.geocode.return_value = (32.794, 34.9896)
    news_repository.exists_by_source_url.return_value = False
    news_repository.save_report.side_effect = lambda report: SaveNewsReportResult(
        report=report,
        is_duplicate=False,
    )


def test_relative_log_file_resolves_to_project_root():
    log_path = NewsMonitoringAgent._resolve_log_path("logs/news_monitoring_agent.log")

    assert log_path == PROJECT_ROOT / "logs" / "news_monitoring_agent.log"


def test_file_logging_can_be_disabled_for_tests(tmp_path):
    config_path = tmp_path / "news_config.yaml"
    log_path = tmp_path / "news_monitoring_agent.log"
    config_path.write_text(
        "\n".join(
            [
                "scraping:",
                "  interval_seconds: 300",
                "logging:",
                "  level: INFO",
                f"  file: \"{log_path.as_posix()}\"",
            ]
        ),
        encoding="utf-8",
    )

    NewsMonitoringAgent(
        config_path=str(config_path),
        rss_fetcher=Mock(),
        text_processor=Mock(),
        geocoder=Mock(),
        news_repository=Mock(spec=NewsRepository),
        enable_file_logging=False,
    )

    assert not log_path.exists()


def test_valid_relevant_article_is_saved():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    rss_fetcher.fetch_all.return_value = [make_entry()]
    _wire_success(text_processor, geocoder, news_repository)
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 1
    news_repository.save_report.assert_called_once()
    report = news_repository.save_report.call_args.args[0]
    assert isinstance(report, WildfireReport)
    assert report.source_url == "https://example.com/news/wildfire-1"
    assert report.source_feed == "Example Feed"
    assert report.location_name == "Haifa"
    assert report.latitude == 32.794
    assert report.longitude == 34.9896
    assert report.published_at == EXPECTED_PUBLISHED_AT
    assert isinstance(report.fetched_at, datetime)
    assert report.fetched_at.tzinfo is not None


def test_irrelevant_article_is_not_saved():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    rss_fetcher.fetch_all.return_value = [make_entry()]
    text_processor.is_relevant.return_value = False
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 0
    news_repository.exists_by_source_url.assert_not_called()
    news_repository.save_report.assert_not_called()
    text_processor.extract_location.assert_not_called()
    geocoder.geocode.assert_not_called()


def test_duplicate_article_is_skipped_before_external_processing():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    rss_fetcher.fetch_all.return_value = [make_entry()]
    text_processor.is_relevant.return_value = True
    news_repository.exists_by_source_url.return_value = True
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 0
    news_repository.save_report.assert_not_called()
    text_processor.extract_location.assert_not_called()
    geocoder.geocode.assert_not_called()


def test_duplicate_report_returned_from_repository_is_not_counted_saved():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    rss_fetcher.fetch_all.return_value = [make_entry()]
    _wire_success(text_processor, geocoder, news_repository)
    news_repository.save_report.return_value = SaveNewsReportResult(
        report=Mock(spec=WildfireReport),
        is_duplicate=True,
    )
    news_repository.save_report.side_effect = None
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 0


def test_missing_location_still_saves_report_with_null_coordinates():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    rss_fetcher.fetch_all.return_value = [make_entry()]
    _wire_success(text_processor, geocoder, news_repository)
    text_processor.extract_location.return_value = None
    geocoder.geocode.return_value = (None, None)
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 1
    report = news_repository.save_report.call_args.args[0]
    assert report.location_name is None
    assert report.latitude is None
    assert report.longitude is None


def test_missing_publication_time_saves_none():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    rss_fetcher.fetch_all.return_value = [make_entry(published="")]
    _wire_success(text_processor, geocoder, news_repository)
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 1
    report = news_repository.save_report.call_args.args[0]
    assert report.published_at is None


def test_malformed_publication_time_does_not_stop_other_articles():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    first = make_entry(link="https://example.com/news/first", published="not a date")
    second = make_entry(link="https://example.com/news/second")
    rss_fetcher.fetch_all.return_value = [first, second]
    _wire_success(text_processor, geocoder, news_repository)
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert saved_count == 2
    first_report = news_repository.save_report.call_args_list[0].args[0]
    second_report = news_repository.save_report.call_args_list[1].args[0]
    assert first_report.published_at is None
    assert second_report.published_at == EXPECTED_PUBLISHED_AT


def test_repository_failure_does_not_stop_other_articles():
    rss_fetcher = Mock()
    text_processor = Mock()
    geocoder = Mock()
    news_repository = Mock(spec=NewsRepository)
    first = make_entry(link="https://example.com/news/first")
    second = make_entry(link="https://example.com/news/second")
    rss_fetcher.fetch_all.return_value = [first, second]
    _wire_success(text_processor, geocoder, news_repository)
    news_repository.save_report.side_effect = [
        NewsRepositoryError("DB write failed"),
        SaveNewsReportResult(report=Mock(spec=WildfireReport), is_duplicate=False),
    ]
    agent = make_agent(rss_fetcher, text_processor, geocoder, news_repository)

    saved_count = agent.run_once()

    assert news_repository.save_report.call_count == 2
    assert saved_count == 1
