"""Data model for a single wildfire news report."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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

    def __post_init__(self) -> None:
        if self.published_at is not None and not isinstance(self.published_at, datetime):
            raise ValueError(f"published_at must be a datetime or None, got {self.published_at!r}")

        if not isinstance(self.fetched_at, datetime):
            raise ValueError(f"fetched_at must be a datetime, got {self.fetched_at!r}")
