"""Normalized inputs for pure active wildfire severity calculation.

Severity is calculated only after active-fire detection has established that a
fire exists. The model intentionally contains scalar domain inputs only: no
FireEvent, repository rows, weather objects, satellite objects, timestamps,
runtime metadata, or persistence identifiers.

FRP is the dominant current-fire intensity input. Wind and humidity describe
current environmental fire behavior. Vegetation fuel is optional in Severity
v1; missing vegetation means unavailable, not zero fuel. Terrain is
intentionally excluded from Severity v1 and may be used later by spread
prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class FireSeverityInput:
    """Complete normalized input required by the pure severity calculator."""

    frp_mw: float
    wind_speed_kmh: float
    relative_humidity_pct: float
    vegetation_fuel_score: float | None = None

    def __post_init__(self) -> None:
        self._validate_finite_number("frp_mw", self.frp_mw)
        self._validate_finite_number("wind_speed_kmh", self.wind_speed_kmh)
        self._validate_finite_number("relative_humidity_pct", self.relative_humidity_pct)

        if self.frp_mw < 0:
            raise ValueError(f"frp_mw must not be negative, got {self.frp_mw!r}")
        if self.wind_speed_kmh < 0:
            raise ValueError(f"wind_speed_kmh must not be negative, got {self.wind_speed_kmh!r}")
        if not 0 <= self.relative_humidity_pct <= 100:
            raise ValueError(
                "relative_humidity_pct must be within [0, 100], "
                f"got {self.relative_humidity_pct!r}"
            )

        if self.vegetation_fuel_score is not None:
            self._validate_finite_number("vegetation_fuel_score", self.vegetation_fuel_score)
            if not 0 <= self.vegetation_fuel_score <= 1:
                raise ValueError(
                    "vegetation_fuel_score must be within [0, 1], "
                    f"got {self.vegetation_fuel_score!r}"
                )

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
