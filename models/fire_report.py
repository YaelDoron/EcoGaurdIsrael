"""Data model for a single wildfire news report."""
from dataclasses import dataclass


@dataclass
class WildfireReport:
    source_url: str # The URL of the report.
    source_feed: str # The report source.
    title: str # The title of the report.
    summary: str # A brief summary of the report.
    location_name: str | None # The name of the location where the wildfire occurred.
    latitude: float | None # The latitude of the wildfire's location.
    longitude: float | None # The longitude of the wildfire's location.
    published_at: str # The date and time when the report was published.
    fetched_at: str # The date and time when the report was fetched by the system.
