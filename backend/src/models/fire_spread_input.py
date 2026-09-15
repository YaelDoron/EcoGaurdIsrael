"""Normalized inputs for the pure wildfire-spread cellular-automata calculator.

The model intentionally contains scalar/enum domain inputs only: no
FireEvent, FireSeverityAssessment, WeatherObservation, repository rows,
database ids, vegetation_fuel_score, or raw Copernicus objects. Retrieving
and converting those into this contract is a Task 5 concern.

Unit and convention contract (see backend/docs/fire_spread_prediction.md,
Scientific Verification Details, for the full sourcing):
- `wind_speed_kmh`: kilometers per hour, non-negative.
- `wind_direction_deg`: degrees clockwise from true north, the standard
  meteorological convention -- the direction FROM which the wind blows
  (matches `WeatherObservation.wind_direction`'s own convention/range).
- `fuel_moisture_percent`: a fine-fuel moisture estimate as a percentage
  (e.g. EcoGuard's existing FFWI equilibrium-moisture-content estimate).
- `fuel_class`: the resolved PROPAGATOR vegetation class, applied
  homogeneously across the whole prediction grid (EcoGuard V1 simplification
  -- see fire_spread_prediction.md, "Spatial vegetation limitation").
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_prediction import SUPPORTED_HORIZON_MINUTES


@dataclass(frozen=True)
class FireSpreadInput:
    """Complete normalized input required by the pure spread calculator."""

    origin_latitude: float
    origin_longitude: float
    wind_speed_kmh: float
    wind_direction_deg: float
    fuel_moisture_percent: float
    fuel_class: FireSpreadFuelClass
    horizon_minutes: int

    def __post_init__(self) -> None:
        self._validate_finite_range("origin_latitude", self.origin_latitude, -90, 90)
        self._validate_finite_range("origin_longitude", self.origin_longitude, -180, 180)

        self._validate_finite_number("wind_speed_kmh", self.wind_speed_kmh)
        if self.wind_speed_kmh < 0:
            raise ValueError(f"wind_speed_kmh must not be negative, got {self.wind_speed_kmh!r}")

        self._validate_finite_range("wind_direction_deg", self.wind_direction_deg, 0, 360)

        self._validate_finite_number("fuel_moisture_percent", self.fuel_moisture_percent)
        if self.fuel_moisture_percent < 0:
            raise ValueError(
                f"fuel_moisture_percent must not be negative, got {self.fuel_moisture_percent!r}"
            )

        if not isinstance(self.fuel_class, FireSpreadFuelClass):
            raise ValueError(f"fuel_class must be a FireSpreadFuelClass, got {self.fuel_class!r}")

        if isinstance(self.horizon_minutes, bool) or not isinstance(self.horizon_minutes, int):
            raise ValueError(f"horizon_minutes must be an integer, got {self.horizon_minutes!r}")
        if self.horizon_minutes not in SUPPORTED_HORIZON_MINUTES:
            raise ValueError(
                f"horizon_minutes must be one of {SUPPORTED_HORIZON_MINUTES}, got {self.horizon_minutes!r}"
            )

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")

    @classmethod
    def _validate_finite_range(cls, field_name: str, value: object, minimum: float, maximum: float) -> None:
        cls._validate_finite_number(field_name, value)
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")
