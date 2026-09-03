"""EcoGuard's internal representation of a single weather observation.

Independent of the raw IMS channel/response shape (that translation lives in
the mapper layer) and independent of any future persistence technology.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WeatherObservation:
    """A single weather reading for a station at a point in time.

    `station_external_id` and `timestamp` are mandatory - without them an
    observation cannot be considered valid. Every meteorological measurement
    may be `None` because not every station measures every channel, some
    measurements may temporarily be invalid, and IMS may return nulls.
    """

    station_external_id: int
    timestamp: datetime

    temperature: float | None = None
    relative_humidity: float | None = None
    wind_speed: float | None = None
    wind_direction: float | None = None
    wind_gust: float | None = None
    rainfall: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.station_external_id, int)
            or isinstance(self.station_external_id, bool)
            or self.station_external_id <= 0
        ):
            raise ValueError(
                f"station_external_id must be a positive integer, got {self.station_external_id!r}"
            )

        if not isinstance(self.timestamp, datetime):
            raise ValueError(f"timestamp must be a datetime, got {self.timestamp!r}")

        if self.relative_humidity is not None and not (0 <= self.relative_humidity <= 100):
            raise ValueError(f"relative_humidity must be within [0, 100], got {self.relative_humidity!r}")

        if self.wind_direction is not None and not (0 <= self.wind_direction <= 360):
            raise ValueError(f"wind_direction must be within [0, 360], got {self.wind_direction!r}")

        if self.wind_speed is not None and self.wind_speed < 0:
            raise ValueError(f"wind_speed must not be negative, got {self.wind_speed!r}")

        if self.wind_gust is not None and self.wind_gust < 0:
            raise ValueError(f"wind_gust must not be negative, got {self.wind_gust!r}")

        if self.rainfall is not None and self.rainfall < 0:
            raise ValueError(f"rainfall must not be negative, got {self.rainfall!r}")
