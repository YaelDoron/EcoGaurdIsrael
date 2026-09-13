"""Prepare normalized weather input for fire-danger calculation."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import math

from src.models import (
    AssessmentArea,
    FireDangerInput,
    FireDangerInputResult,
    FireDangerInputStatus,
)
from src.repositories.weather_repository import StoredWeatherObservation, WeatherRepository
from src.services.fire_danger.fire_danger_input_config import (
    MAX_WEATHER_AGE_MINUTES,
    MIN_VALID_STATIONS,
)

_EARTH_RADIUS_KM = 6371.0088


class FireDangerInputService:
    """Build FFWI input data from recent weather observations in an area.

    This service selects and aggregates weather inputs only. It does not
    calculate FFWI, classify danger levels, persist assessments, inspect
    satellite/news data, or use simulation metadata.
    """

    def __init__(self, weather_repository: WeatherRepository | None = None) -> None:
        self._weather_repository = weather_repository or WeatherRepository()

    def build_input(self, area: AssessmentArea, as_of: datetime) -> FireDangerInputResult:
        """Build a FireDangerInput for an area at a timezone-aware instant."""
        if not isinstance(area, AssessmentArea):
            raise ValueError(f"area must be an AssessmentArea, got {area!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")

        earliest = as_of - timedelta(minutes=MAX_WEATHER_AGE_MINUTES)
        candidates = self._weather_repository.get_recent_observations_for_area_candidates(
            latitude=area.latitude,
            longitude=area.longitude,
            radius_km=area.radius_km,
            start_time=earliest,
            end_time=as_of,
        )
        selected = self._select_latest_valid_observation_per_station(area, as_of, candidates)

        if len(selected) < MIN_VALID_STATIONS:
            return FireDangerInputResult(
                status=FireDangerInputStatus.INSUFFICIENT_DATA,
                input_data=None,
                observation_ids=(),
                station_ids=(),
            )

        temperature_c = sum(record.observation.temperature for record in selected) / len(selected)
        relative_humidity_pct = sum(record.observation.relative_humidity for record in selected) / len(
            selected
        )
        wind_speed_kmh = sum(record.observation.wind_speed for record in selected) / len(selected)

        return FireDangerInputResult(
            status=FireDangerInputStatus.READY,
            input_data=FireDangerInput(
                temperature_c=temperature_c,
                relative_humidity_pct=relative_humidity_pct,
                wind_speed_kmh=wind_speed_kmh,
            ),
            observation_ids=tuple(record.observation_id for record in selected),
            station_ids=tuple(record.station_id for record in selected),
        )

    def _select_latest_valid_observation_per_station(
        self,
        area: AssessmentArea,
        as_of: datetime,
        candidates: list[StoredWeatherObservation],
    ) -> tuple[StoredWeatherObservation, ...]:
        sorted_candidates = sorted(
            candidates,
            key=lambda record: (
                record.station_id,
                -_timestamp_sort_value(_ensure_aware(record.observation.timestamp)),
                record.observation_id,
            ),
        )
        selected_by_station: OrderedDict[int, StoredWeatherObservation] = OrderedDict()

        for record in sorted_candidates:
            if record.station_id in selected_by_station:
                continue
            if not _station_is_in_area(area, record):
                continue
            if not _observation_is_fresh(record.observation.timestamp, as_of):
                continue
            if not _observation_has_required_fields(record):
                continue
            selected_by_station[record.station_id] = record

        return tuple(selected_by_station.values())


def haversine_distance_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    """Return Haversine distance in kilometers between two coordinates."""
    first_latitude_rad = math.radians(first_latitude)
    second_latitude_rad = math.radians(second_latitude)
    latitude_delta = math.radians(second_latitude - first_latitude)
    longitude_delta = math.radians(second_longitude - first_longitude)

    a = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude_rad)
        * math.cos(second_latitude_rad)
        * math.sin(longitude_delta / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_KM * c


def _station_is_in_area(area: AssessmentArea, record: StoredWeatherObservation) -> bool:
    distance = haversine_distance_km(
        area.latitude,
        area.longitude,
        record.station.latitude,
        record.station.longitude,
    )
    return distance <= area.radius_km


def _observation_is_fresh(timestamp: datetime, as_of: datetime) -> bool:
    observed_at = _ensure_aware(timestamp)
    age = as_of - observed_at
    return timedelta(0) <= age <= timedelta(minutes=MAX_WEATHER_AGE_MINUTES)


def _observation_has_required_fields(record: StoredWeatherObservation) -> bool:
    observation = record.observation
    return (
        observation.temperature is not None
        and observation.relative_humidity is not None
        and observation.wind_speed is not None
    )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _timestamp_sort_value(value: datetime) -> float:
    return value.timestamp()
