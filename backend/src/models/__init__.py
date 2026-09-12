"""EcoGuard internal domain models (plain dataclasses, no persistence)."""

from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.models.fire_report import WildfireReport

__all__ = ["WeatherStation", "WeatherObservation", "SatelliteHotspot", "WildfireReport"]
