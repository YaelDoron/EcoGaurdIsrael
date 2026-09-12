"""Shared simulation location definitions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationLocation:
    """A named geographic anchor shared by simulated data sources."""

    name: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if (
            not isinstance(self.latitude, (int, float))
            or isinstance(self.latitude, bool)
            or not (-90 <= self.latitude <= 90)
        ):
            raise ValueError(f"latitude must be within [-90, 90], got {self.latitude!r}")
        if (
            not isinstance(self.longitude, (int, float))
            or isinstance(self.longitude, bool)
            or not (-180 <= self.longitude <= 180)
        ):
            raise ValueError(f"longitude must be within [-180, 180], got {self.longitude!r}")
