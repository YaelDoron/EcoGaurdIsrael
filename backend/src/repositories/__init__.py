"""Persistence layer bridging Task 2 domain models and SQLAlchemy ORM models."""

from src.repositories.exceptions import (
    SatelliteHotspotRepositoryError,
    WeatherRepositoryError,
    WeatherStationNotStoredError,
)
from src.repositories.satellite_hotspot_repository import SaveHotspotResult, SatelliteHotspotRepository
from src.repositories.weather_repository import SaveObservationResult, WeatherRepository

__all__ = [
    "WeatherRepository",
    "SaveObservationResult",
    "SatelliteHotspotRepository",
    "SaveHotspotResult",
    "WeatherRepositoryError",
    "WeatherStationNotStoredError",
    "SatelliteHotspotRepositoryError",
]
