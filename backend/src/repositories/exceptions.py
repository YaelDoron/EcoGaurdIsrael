"""Exceptions raised by the persistence (repository) layer."""


class WeatherRepositoryError(Exception):
    """Base exception for all weather repository errors."""


class WeatherStationNotStoredError(WeatherRepositoryError):
    """Raised when an observation references a station that has not been saved yet."""


class SatelliteHotspotRepositoryError(Exception):
    """Base exception for satellite hotspot repository errors."""
