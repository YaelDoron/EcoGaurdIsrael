"""EcoGuard internal domain models (plain dataclasses, no persistence)."""

from src.models.assessment_area import AssessmentArea
from src.models.fire_danger_calculation import FireDangerCalculation
from src.models.fire_danger_input import FireDangerInput
from src.models.fire_danger_input_result import FireDangerInputResult
from src.models.fire_danger_input_status import FireDangerInputStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_report import WildfireReport
from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

__all__ = [
    "WeatherStation",
    "WeatherObservation",
    "SatelliteHotspot",
    "WildfireReport",
    "FireDangerInput",
    "FireDangerCalculation",
    "FireDangerLevel",
    "AssessmentArea",
    "FireDangerInputResult",
    "FireDangerInputStatus",
]
