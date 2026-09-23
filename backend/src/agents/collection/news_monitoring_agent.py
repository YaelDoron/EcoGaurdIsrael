"""NewsMonitoringAgent: fetches wildfire news and persists new reports."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from src.config.config import load_config
from src.models.fire_report import WildfireReport
from src.repositories.exceptions import NewsRepositoryError
from src.repositories.news_repository import NewsRepository

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class NewsMonitoringAgent:
    """Coordinates one RSS -> filter -> location -> geocode -> persist cycle."""

    def __init__(
        self,
        config_path: str | None = None,
        rss_fetcher: Any | None = None,
        text_processor: Any | None = None,
        geocoder: Any | None = None,
        news_repository: NewsRepository | None = None,
        enable_file_logging: bool = True,
    ) -> None:
        self.config = load_config(config_path) if config_path else load_config()
        self._setup_logging(enable_file_logging=enable_file_logging)

        if rss_fetcher is None or text_processor is None:
            from src.external.news.news_client import RSSFetcher, TextProcessor

        if geocoder is None:
            from src.external.geocoding.geocoding_client import Geocoder

        self.rss_fetcher = rss_fetcher or RSSFetcher(self.config["rss_feeds"])
        self.text_processor = text_processor or TextProcessor(self.config["keywords"], self.config["llm"])
        self.geocoder = geocoder or Geocoder(self.config["geocoding"])
        self.news_repository = news_repository or NewsRepository()
        self.interval_seconds = self.config["scraping"]["interval_seconds"]

    # Sets up logging based on the configuration, allowing for console and optional file logging.
    def _setup_logging(self, *, enable_file_logging: bool = True) -> None:
        log_config = self.config.get("logging", {})
        handlers = [logging.StreamHandler()]
        log_file = log_config.get("file")
        if enable_file_logging and log_file:
            log_path = self._resolve_log_path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=handlers,
            force=True,
        )

    @staticmethod
    def _resolve_log_path(log_file: str) -> Path:
        path = Path(log_file).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    # The cycle of fetching, filtering, extracting, geocoding, and storing is encapsulated in this method.
    def run_once(self) -> int:
        """Runs a single fetch-filter-extract-geocode-store cycle. Returns the number
        of new reports saved."""
        # Fetch all entries from the configured RSS feeds
        entries = self.rss_fetcher.fetch_all()
        # Filter entries based on relevance to the configured keywords using the TextProcessor
        relevant_entries = [
            e for e in entries if self.text_processor.is_relevant(e["title"], e["summary"])
        ]
        logger.info("%d/%d entries matched keyword filter", len(relevant_entries), len(entries))

        saved_count = 0
        for entry in relevant_entries:
            source_url = entry["link"]
            if not source_url or self.news_repository.exists_by_source_url(source_url):
                continue

            analysis = self.text_processor.analyze(entry["title"], entry["summary"])
            latitude, longitude = self.geocoder.geocode(analysis.location_name)

            # Create a WildfireReport object with the extracted and geocoded information
            # The object is saved to the database.
            report = WildfireReport(
                source_url=source_url,
                source_feed=entry["source_feed"],
                title=entry["title"],
                summary=entry["summary"],
                location_name=analysis.location_name,
                latitude=latitude,
                longitude=longitude,
                published_at=self._parse_published_at(entry.get("published")),
                fetched_at=datetime.now(timezone.utc),
                wildfire_signal_strength=analysis.wildfire_signal_strength,
            )

            try:
                save_result = self.news_repository.save_report(report)
            except NewsRepositoryError:
                logger.exception("Failed to save wildfire news report: %s", report.source_url)
                continue

            if not save_result.is_duplicate:
                saved_count += 1
                logger.info(
                    "Saved report: '%s...' -> location=%s (%s, %s), signal=%s",
                    report.title[:60],
                    analysis.location_name,
                    latitude,
                    longitude,
                    analysis.wildfire_signal_strength,
                )

        logger.info("Cycle complete: %d new reports saved", saved_count)
        return saved_count

    # Continuously runs the monitoring cycle at the configured inteval.
    def run_forever(self) -> None:
        logger.info(
            "NewsMonitoringAgent starting, polling every %d seconds", self.interval_seconds
        )
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("Unhandled error during monitoring cycle")
            time.sleep(self.interval_seconds)

    @staticmethod
    def _parse_published_at(value: Any) -> datetime | None:
        """Parse an optional RSS publication timestamp into an aware datetime."""
        if not isinstance(value, str) or not value.strip():
            return None

        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed RSS publication timestamp: %r", value)
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
