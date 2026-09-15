"""EcoGuard's internal representation of a fire station.

`id` is the station identifier as reported by the governmental source data,
mirroring how `WeatherStation.external_station_id` tracks IMS's own id -
no separate database identity exists at the domain layer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FireStation:
    """A fire station known to EcoGuard."""

    id: int | str
    name: str
    latitude: float
    longitude: float
    station_type: str | None = None
    address: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, (int, str))
            or isinstance(self.id, bool)
            or (isinstance(self.id, int) and self.id <= 0)
            or (isinstance(self.id, str) and not self.id.strip())
        ):
            raise ValueError(f"id must be a positive int or non-empty string, got {self.id!r}")

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

        if self.station_type is not None and not isinstance(self.station_type, str):
            raise ValueError(f"station_type must be a string or None, got {self.station_type!r}")

        if self.address is not None and not isinstance(self.address, str):
            raise ValueError(f"address must be a string or None, got {self.address!r}")
