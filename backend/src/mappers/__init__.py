"""Mappers translating raw external API data into EcoGuard's internal models."""

from src.mappers.exceptions import (
    InvalidSatelliteDetectionTimeError,
    InvalidWeatherTimestampError,
    MissingRequiredSatelliteFieldError,
    MissingRequiredWeatherFieldError,
    SatelliteHotspotMappingError,
    WeatherMappingError,
)
from src.mappers.satellite_hotspot_mapper import SatelliteHotspotMapper
from src.mappers.vegetation_mapper import VegetationMapper
from src.mappers.weather_mapper import WeatherMapper

__all__ = [
    "WeatherMapper",
    "SatelliteHotspotMapper",
    "VegetationMapper",
    "WeatherMappingError",
    "MissingRequiredWeatherFieldError",
    "InvalidWeatherTimestampError",
    "SatelliteHotspotMappingError",
    "MissingRequiredSatelliteFieldError",
    "InvalidSatelliteDetectionTimeError",
]
