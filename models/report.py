"""Data model for a single wildfire news report."""
from dataclasses import dataclass


@dataclass
class WildfireReport:
    source_url: str
    source_feed: str
    title: str
    summary: str
    location_name: str | None
    latitude: float | None
    longitude: float | None
    published_at: str
    fetched_at: str
