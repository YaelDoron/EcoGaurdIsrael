"""Exceptions raised while mapping raw external API data into EcoGuard models."""


class WeatherMappingError(ValueError):
    """Base exception for all weather-mapping errors."""


class MissingRequiredWeatherFieldError(WeatherMappingError):
    """Raised when a mandatory raw IMS field (identity data) is missing or invalid."""


class InvalidWeatherTimestampError(WeatherMappingError):
    """Raised when a raw IMS observation timestamp cannot be parsed."""


class SatelliteHotspotMappingError(ValueError):
    """Base exception for satellite hotspot mapping errors."""


class MissingRequiredSatelliteFieldError(SatelliteHotspotMappingError):
    """Raised when a mandatory raw FIRMS detection field is missing or invalid."""


class InvalidSatelliteDetectionTimeError(SatelliteHotspotMappingError):
    """Raised when FIRMS acquisition date/time fields cannot be parsed."""
