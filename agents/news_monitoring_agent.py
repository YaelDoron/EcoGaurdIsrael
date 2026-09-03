"""NewsMonitoringAgent: fetches wildfire news from RSS feeds, extracts and geocodes
the location via LLM, and persists new reports. This agent has no other responsibility
within the larger wildfire management system."""
import logging
import time
from datetime import datetime, timezone

from core.config import load_config
from core.geocoder import Geocoder
from core.rss_fetcher import RSSFetcher
from core.storage_manager import StorageManager
from core.text_processor import TextProcessor
from models.report import WildfireReport

logger = logging.getLogger(__name__)


class NewsMonitoringAgent:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path) if config_path else load_config()
        self._setup_logging()

        self.rss_fetcher = RSSFetcher(self.config["rss_feeds"])
        self.text_processor = TextProcessor(self.config["keywords"], self.config["llm"])
        self.geocoder = Geocoder(self.config["geocoding"])
        self.storage = StorageManager(self.config["storage"]["db_path"])
        self.interval_seconds = self.config["scraping"]["interval_seconds"]

    def _setup_logging(self) -> None:
        log_config = self.config.get("logging", {})
        handlers = [logging.StreamHandler()]
        log_file = log_config.get("file")
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=handlers,
            force=True,
        )

    def run_once(self) -> int:
        """Runs a single fetch-filter-extract-geocode-store cycle. Returns the number
        of new reports saved."""
        entries = self.rss_fetcher.fetch_all()
        relevant_entries = [
            e for e in entries if self.text_processor.is_relevant(e["title"], e["summary"])
        ]
        logger.info("%d/%d entries matched keyword filter", len(relevant_entries), len(entries))

        saved_count = 0
        for entry in relevant_entries:
            source_url = entry["link"]
            if not source_url or self.storage.exists(source_url):
                continue

            location_name = self.text_processor.extract_location(entry["title"], entry["summary"])
            latitude, longitude = self.geocoder.geocode(location_name)

            report = WildfireReport(
                source_url=source_url,
                source_feed=entry["source_feed"],
                title=entry["title"],
                summary=entry["summary"],
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                published_at=entry.get("published", ""),
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )

            if self.storage.save_report(report):
                saved_count += 1
                logger.info(
                    "Saved report: '%s...' -> location=%s (%s, %s)",
                    report.title[:60],
                    location_name,
                    latitude,
                    longitude,
                )

        logger.info("Cycle complete: %d new reports saved", saved_count)
        return saved_count

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
