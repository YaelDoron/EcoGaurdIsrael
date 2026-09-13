"""Persistence layer bridging Task 2 domain models and SQLAlchemy ORM models."""

from src.repositories.exceptions import (
    FireDangerAssessmentRepositoryError,
    NewsRepositoryError,
    SatelliteHotspotRepositoryError,
    WeatherRepositoryError,
    WeatherStationNotStoredError,
)
from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.repositories.news_repository import NewsRepository, SaveNewsReportResult
from src.repositories.satellite_hotspot_repository import SaveHotspotResult, SatelliteHotspotRepository
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
    "NewsRepository",
    "SaveNewsReportResult",
    "FireDangerAssessmentRepository",
    "StoredFireDangerAssessment",
    "WeatherRepositoryError",
    "WeatherStationNotStoredError",
    "SatelliteHotspotRepositoryError",
    "NewsRepositoryError",
    "FireDangerAssessmentRepositoryError",
]
