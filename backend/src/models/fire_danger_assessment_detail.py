"""Read-only single-assessment detail DTOs (Task A4, Part 7).

Describes one already-persisted FireDangerAssessment together with the
persisted weather-observation trace it was calculated from
(fire_danger_assessment_weather_inputs, see
src/repositories/fire_danger_assessment_repository.py). This is a strict
historical read: no FFWI recalculation and no live weather fetch happens
here or in FireDangerQueryService - `score`/`level`/`assessed_at` are copied
unchanged from the persisted row, and `weather_inputs` are the exact
WeatherObservation rows FireDangerAssessmentRepository already traced at
save time, re-materialized via WeatherRepository.get_observations_by_ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.optimization_validation import validate_non_empty_string, validate_positive_int


@dataclass(frozen=True)
class FireDangerAssessmentWeatherInputSummary:
    """One persisted weather observation traced as input to an assessment."""

    observation_id: int
    station_external_id: int
    station_name: str
    observed_at: datetime

    def __post_init__(self) -> None:
        validate_positive_int("observation_id", self.observation_id)
        validate_positive_int("station_external_id", self.station_external_id)
        validate_non_empty_string("station_name", self.station_name)
        _validate_aware_or_naive_datetime("observed_at", self.observed_at)


@dataclass(frozen=True)
class FireDangerAssessmentDetail:
    """One persisted fire-danger assessment with area metadata and its weather-input trace."""

    assessment_id: int
    area_id: str
    area_name: str
    area_latitude: float
    area_longitude: float
    area_radius_km: float
    status: FireDangerAssessmentStatus
    score: float | None
    level: FireDangerLevel | None
    assessed_at: datetime
    methodology: str
    methodology_version: str
    weather_inputs: tuple[FireDangerAssessmentWeatherInputSummary, ...]

    def __post_init__(self) -> None:
        validate_positive_int("assessment_id", self.assessment_id)
        validate_non_empty_string("area_id", self.area_id)
        validate_non_empty_string("area_name", self.area_name)
        _validate_coordinate("area_latitude", self.area_latitude, -90, 90)
        _validate_coordinate("area_longitude", self.area_longitude, -180, 180)
        _validate_finite_number("area_radius_km", self.area_radius_km)
        if self.area_radius_km <= 0:
            raise ValueError(f"area_radius_km must be greater than 0, got {self.area_radius_km!r}")
        if not isinstance(self.status, FireDangerAssessmentStatus):
            raise ValueError(f"status must be a FireDangerAssessmentStatus, got {self.status!r}")
        if self.score is not None:
            _validate_finite_number("score", self.score)
        if self.level is not None and not isinstance(self.level, FireDangerLevel):
            raise ValueError(f"level must be a FireDangerLevel or None, got {self.level!r}")
        _validate_aware_datetime("assessed_at", self.assessed_at)
        validate_non_empty_string("methodology", self.methodology)
        validate_non_empty_string("methodology_version", self.methodology_version)
        weather_inputs = _coerce_tuple("weather_inputs", self.weather_inputs)
        for item in weather_inputs:
            if not isinstance(item, FireDangerAssessmentWeatherInputSummary):
                raise ValueError(
                    f"weather_inputs must contain FireDangerAssessmentWeatherInputSummary entries, got {item!r}"
                )
        object.__setattr__(self, "weather_inputs", weather_inputs)


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_finite_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
    _validate_finite_number(field_name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")


def _validate_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")


def _validate_aware_or_naive_datetime(field_name: str, value: object) -> None:
    # WeatherObservationDB.timestamp is a naive DateTime column (see
    # src/database/models/weather_observation_db.py) - unlike assessed_at,
    # observed_at is not required to carry tzinfo, matching
    # WeatherObservation's own domain model.
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime, got {value!r}")
