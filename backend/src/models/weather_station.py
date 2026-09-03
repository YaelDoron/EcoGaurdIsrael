"""EcoGuard's internal representation of a weather station.

This model is independent of the raw IMS API shape (that translation lives in
the mapper layer) and independent of any future persistence technology - no
database id exists yet; that is added when SQLAlchemy persistence is
introduced in a later task.
"""
from dataclasses import dataclass


@dataclass
class WeatherStation:
    """A weather station known to EcoGuard.

    `external_station_id` is the station identifier as reported by IMS.
    """

    external_station_id: int
    name: str
    latitude: float
    longitude: float
    region_id: int | None = None
    active: bool | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.external_station_id, int)
            or isinstance(self.external_station_id, bool)
            or self.external_station_id <= 0
        ):
            raise ValueError(
                f"external_station_id must be a positive integer, got {self.external_station_id!r}"
            )

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

        if self.region_id is not None and (
            not isinstance(self.region_id, int) or isinstance(self.region_id, bool) or self.region_id <= 0
        ):
            raise ValueError(f"region_id must be a positive integer or None, got {self.region_id!r}")

        if self.active is not None and not isinstance(self.active, bool):
            raise ValueError(f"active must be a boolean or None, got {self.active!r}")
