"""EcoGuard's internal representation of a satellite thermal anomaly detection.

This is a plain domain dataclass. It does not represent a confirmed wildfire
and has no persistence/database identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SatelliteHotspot:
    """A satellite-detected thermal anomaly / active-fire detection.

    `location_name` is optional, trustworthy provenance: it is set ONLY by
    the demo simulation's own SatelliteDataGenerator, to the exact canonical
    SimulationLocation name that produced this hotspot - never a guess, and
    never set by real FIRMS ingestion (SatelliteHotspotMapper never
    populates it, so a real detection's value is always None). This lets a
    FireDetectionAgent-created FireEvent inherit a reliable location without
    any read-side geography guessing.
    """

    latitude: float
    longitude: float
    detected_at: datetime

    confidence: str | None = None
    frp: float | None = None
    brightness: float | None = None

    satellite: str | None = None
    instrument: str | None = None
    day_night: str | None = None
    location_name: str | None = None

    def __post_init__(self) -> None:
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

        if not isinstance(self.detected_at, datetime):
            raise ValueError(f"detected_at must be a datetime, got {self.detected_at!r}")

        if self.frp is not None and (
            not isinstance(self.frp, (int, float)) or isinstance(self.frp, bool) or self.frp < 0
        ):
            raise ValueError(f"frp must be a non-negative number or None, got {self.frp!r}")

        if self.brightness is not None and (
            not isinstance(self.brightness, (int, float))
            or isinstance(self.brightness, bool)
            or self.brightness < 0
        ):
            raise ValueError(
                f"brightness must be a non-negative number or None, got {self.brightness!r}"
            )

        for field_name in ("confidence", "satellite", "instrument", "day_night", "location_name"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or None, got {value!r}")
