"""Fetches and parses configured RSS feeds."""
import logging
from typing import Any

import feedparser

logger = logging.getLogger(__name__)


class RSSFetcher:
    def __init__(self, feeds: list[dict[str, str]]):
        self.feeds = feeds

    def fetch_all(self) -> list[dict[str, Any]]:
        """Fetch every configured feed and return a flat list of raw entries."""
        entries: list[dict[str, Any]] = []
        for feed in self.feeds:
            name, url = feed["name"], feed["url"]
            try:
                parsed = feedparser.parse(url)
            except Exception:
                logger.exception("Failed to fetch feed '%s' (%s)", name, url)
                continue

            if parsed.bozo and not parsed.entries:
                logger.warning(
                    "Feed '%s' returned no usable entries: %s",
                    name,
                    getattr(parsed, "bozo_exception", "unknown parse error"),
                )
                continue

            for entry in parsed.entries:
                entries.append(
                    {
                        "source_feed": name,
                        "title": entry.get("title", "").strip(),
                        "summary": entry.get("summary", entry.get("description", "")).strip(),
                        "link": entry.get("link", "").strip(),
                        "published": entry.get("published", ""),
                    }
                )
        logger.info("Fetched %d total entries from %d feeds", len(entries), len(self.feeds))
        return entries
