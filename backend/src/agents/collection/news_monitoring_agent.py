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

    # The cycle of fetching, filtering, analyzing, geocoding, translating, and storing is encapsulated in this method.
    def run_once(self) -> int:
        """Runs a single fetch-filter-analyze-geocode-translate-store cycle. Returns the
        number of new reports saved. A failure on one article is logged and that article
        skipped - it never aborts the rest of the cycle."""
        # Fetch all entries from the configured RSS feeds
        entries = self.rss_fetcher.fetch_all()
        # Filter entries based on relevance to the configured keywords using the TextProcessor
        relevant_entries = [e for e in entries if self._is_relevant_entry(e)]
        logger.info("%d/%d entries matched keyword filter", len(relevant_entries), len(entries))

        saved_count = 0
        for entry in relevant_entries:
            try:
                if self._process_entry(entry):
                    saved_count += 1
            except Exception:  # noqa: BLE001 - one bad article must never abort the whole cycle.
                logger.exception(
                    "Skipping wildfire news entry after an unexpected error: %s", self._entry_label(entry)
                )

        logger.info("Cycle complete: %d new reports saved", saved_count)
        return saved_count

    def _is_relevant_entry(self, entry: Any) -> bool:
        """Keyword relevance for one entry; a malformed entry is logged and treated as irrelevant."""
        try:
            return bool(self.text_processor.is_relevant(entry["title"], entry["summary"]))
        except Exception:  # noqa: BLE001 - a malformed entry must not abort the cycle.
            logger.exception("Skipping malformed feed entry (relevance check failed): %s", self._entry_label(entry))
            return False

    def _process_entry(self, entry: dict[str, Any]) -> bool:
        """Analyze, geocode, translate and persist ONE relevant entry.

        Returns True only when a NEW report was saved (False for an already-known URL,
        a duplicate the repository reports, or a repository failure that was logged).
        """
        source_url = entry["link"]
        if not source_url or self.news_repository.exists_by_source_url(source_url):
            return False

        # ONE LLM call returns both the article's location and its wildfire signal
        # strength. TextProcessor.analyze never raises: if the LLM is unavailable it
        # returns location_name=None / wildfire_signal_strength=None ("analysis
        # unavailable", never a fabricated NONE) and the report is still stored.
        analysis = self.text_processor.analyze(entry["title"], entry["summary"])
        # Geocoding uses the ORIGINAL (Hebrew) location name, extracted
        # above - Nominatim resolves Israeli place names most reliably
        # in their native script. Translation happens after, and only
        # changes what gets displayed/persisted, never what got geocoded.
        # Geocoder.geocode never raises: (None, None) on a miss or failure.
        latitude, longitude = self.geocoder.geocode(analysis.location_name)

        title_en, summary_en, location_name_en = self.text_processor.translate_report(
            entry["title"], entry["summary"], analysis.location_name
        )

        # Create a WildfireReport object with the translated text, the geocoded
        # coordinates and the analyzed signal strength. The object is saved to the database.
        report = WildfireReport(
            source_url=source_url,
            source_feed=entry["source_feed"],
            title=title_en,
            summary=summary_en,
            location_name=location_name_en,
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
            return False

        if save_result.is_duplicate:
            return False

        logger.info(
            "Saved report: '%s...' -> location=%s (%s, %s), signal=%s",
            report.title[:60],
            analysis.location_name,
            latitude,
            longitude,
            analysis.wildfire_signal_strength,
        )
        return True

    @staticmethod
    def _entry_label(entry: Any) -> str:
        """A short, safe identifier for logging an entry that may itself be malformed."""
        if isinstance(entry, dict):
            return str(entry.get("link") or entry.get("title") or "<entry without link/title>")[:120]
        return repr(entry)[:120]

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
