"""Exceptions raised while mapping raw IMS JSON into EcoGuard's internal models."""


class WeatherMappingError(ValueError):
    """Base exception for all weather-mapping errors."""


class MissingRequiredWeatherFieldError(WeatherMappingError):
    """Raised when a mandatory raw IMS field (identity data) is missing or invalid."""


class InvalidWeatherTimestampError(WeatherMappingError):
    """Raised when a raw IMS observation timestamp cannot be parsed."""
