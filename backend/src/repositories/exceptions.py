"""Exceptions raised by the persistence (repository) layer."""


class WeatherRepositoryError(Exception):
    """Base exception for all weather repository errors."""


class WeatherStationNotStoredError(WeatherRepositoryError):
    """Raised when an observation references a station that has not been saved yet."""


class SatelliteHotspotRepositoryError(Exception):
    """Base exception for satellite hotspot repository errors."""


class NewsRepositoryError(Exception):
    """Base exception for wildfire news repository errors."""


class FireDangerAssessmentRepositoryError(Exception):
    """Base exception for fire-danger assessment repository errors."""


class FireEventRepositoryError(Exception):
    """Base exception for wildfire event repository errors."""
