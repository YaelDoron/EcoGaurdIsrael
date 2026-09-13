"""Normalized weather inputs for fire-danger calculations."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class FireDangerInput:
    """Input required by the Fosberg Fire Weather Index calculator.

    This model intentionally contains only normalized weather values. The
    calculator does not receive WeatherObservation, repository objects,
    simulation metadata, or external API payloads.
    """

    temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float

    def __post_init__(self) -> None:
        self._validate_finite_number("temperature_c", self.temperature_c)
        self._validate_finite_number("relative_humidity_pct", self.relative_humidity_pct)
        self._validate_finite_number("wind_speed_kmh", self.wind_speed_kmh)

        if not 0 <= self.relative_humidity_pct <= 100:
            raise ValueError(
                "relative_humidity_pct must be within [0, 100], "
                f"got {self.relative_humidity_pct!r}"
            )
        if self.wind_speed_kmh < 0:
            raise ValueError(f"wind_speed_kmh must not be negative, got {self.wind_speed_kmh!r}")

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
