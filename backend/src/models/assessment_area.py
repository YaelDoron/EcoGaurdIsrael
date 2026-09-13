"""Domain model describing an area for fire-danger input selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class AssessmentArea:
    """Geographic area where EcoGuard should assess fire-weather danger."""

    id: str
    name: str
    latitude: float
    longitude: float
    radius_km: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        self._validate_finite_number("latitude", self.latitude)
        self._validate_finite_number("longitude", self.longitude)
        self._validate_finite_number("radius_km", self.radius_km)
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude must be within [-90, 90], got {self.latitude!r}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude must be within [-180, 180], got {self.longitude!r}")
        if self.radius_km <= 0:
            raise ValueError(f"radius_km must be greater than 0, got {self.radius_km!r}")

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
