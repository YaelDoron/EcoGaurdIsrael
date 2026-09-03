"""EcoGuard internal domain models (plain dataclasses, no persistence)."""

from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

__all__ = ["WeatherStation", "WeatherObservation"]
