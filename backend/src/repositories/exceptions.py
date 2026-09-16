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


class FireSeverityAssessmentRepositoryError(Exception):
    """Base exception for fire-severity assessment repository errors."""


class FireSpreadPredictionRepositoryError(Exception):
    """Base exception for wildfire-spread prediction repository errors."""


class ResponseTargetRepositoryError(Exception):
    """Base exception for response-target repository errors."""


class FirefightingResourceRepositoryError(Exception):
    """Base exception for firefighting-resource repository errors."""


class RoutePlanningRepositoryError(Exception):
    """Base exception for routing-run repository errors."""
