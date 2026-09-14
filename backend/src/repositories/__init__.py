"""Persistence layer bridging Task 2 domain models and SQLAlchemy ORM models."""

from src.repositories.exceptions import (
    FireDangerAssessmentRepositoryError,
    FireEventRepositoryError,
    NewsRepositoryError,
    SatelliteHotspotRepositoryError,
    WeatherRepositoryError,
    WeatherStationNotStoredError,
)
from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.news_repository import NewsRepository, SaveNewsReportResult, StoredWildfireReport
from src.repositories.satellite_hotspot_repository import (
    SaveHotspotResult,
    SatelliteHotspotRepository,
    StoredSatelliteHotspot,
)
from src.repositories.weather_repository import (
    SaveObservationResult,
    StoredWeatherObservation,
    WeatherRepository,
)

__all__ = [
    "WeatherRepository",
    "SaveObservationResult",
    "StoredWeatherObservation",
    "SatelliteHotspotRepository",
    "SaveHotspotResult",
    "StoredSatelliteHotspot",
    "NewsRepository",
    "SaveNewsReportResult",
    "StoredWildfireReport",
    "FireDangerAssessmentRepository",
    "StoredFireDangerAssessment",
    "FireEventRepository",
    "StoredFireEvent",
    "WeatherRepositoryError",
    "WeatherStationNotStoredError",
    "SatelliteHotspotRepositoryError",
    "NewsRepositoryError",
    "FireDangerAssessmentRepositoryError",
    "FireEventRepositoryError",
]
