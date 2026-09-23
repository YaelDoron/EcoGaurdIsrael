"""Data model for a single wildfire news report."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength


@dataclass
class WildfireReport:
    source_url: str # The URL of the report.
    source_feed: str # The report source.
    title: str # The title of the report.
    summary: str # A brief summary of the report.
    location_name: str | None # The name of the location where the wildfire occurred.
    latitude: float | None # The latitude of the wildfire's location.
    longitude: float | None # The longitude of the wildfire's location.
    published_at: datetime | None # The date and time when the report was published.
    fetched_at: datetime # The date and time when the report was fetched by the system.
    # How strongly the article's own text claims an active wildfire (LLM-derived,
    # semantic - not ground truth, not a calibrated probability). None means no
    # reliable analysis is available (including every historical row saved
    # before this field existed) - never fabricated as NONE. Defaults to None
    # so existing callers that do not pass it remain valid.
    wildfire_signal_strength: NewsWildfireSignalStrength | None = None

    def __post_init__(self) -> None:
        if self.published_at is not None and not isinstance(self.published_at, datetime):
            raise ValueError(f"published_at must be a datetime or None, got {self.published_at!r}")

        if not isinstance(self.fetched_at, datetime):
            raise ValueError(f"fetched_at must be a datetime, got {self.fetched_at!r}")

        if self.wildfire_signal_strength is not None and not isinstance(
            self.wildfire_signal_strength, NewsWildfireSignalStrength
        ):
            raise ValueError(
                "wildfire_signal_strength must be a NewsWildfireSignalStrength or None, "
                f"got {self.wildfire_signal_strength!r}"
            )
