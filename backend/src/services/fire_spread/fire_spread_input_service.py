"""Prepare normalized inputs for the pure wildfire-spread CA calculator.

This service performs input retrieval and preparation only. It does not run
fire detection, does not recalculate fire danger or fire severity, does not
call Copernicus or IMS, does not run the CA propagation, and does not
persist a FireSpreadPrediction. See backend/docs/fire_spread_prediction.md
for the reuse-first architectural rule this service implements.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.calculators.fire_danger.ffwi_calculator import _celsius_to_fahrenheit, _equilibrium_moisture_content
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_input_result import FireSpreadInputResult
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_prediction import SUPPORTED_HORIZON_MINUTES
from src.models.weather_observation import WeatherObservation
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.repositories.weather_repository import StoredWeatherObservation, WeatherRepository
from src.services.fire_severity.fire_severity_input_config import MAX_SEVERITY_WEATHER_AGE_MINUTES
from src.utils.geo import haversine_distance_km

_ACTIVE_EVENT_STATUSES = {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}
_INACTIVE_EVENT_STATUSES = {FireEventStatus.RESOLVED, FireEventStatus.DISMISSED}

# IMS's WS channel is documented in meters/second (verified in Task 4A against
# the official IMS API documentation, Appendix C); WeatherMapper persists it
# unconverted. FireSpreadCalculator's verified contract requires km/h
# (fire_spread_prediction.md, Scientific Verification Details). This is the
# single, explicit conversion point for wildfire-spread prediction -- it does
# not touch WeatherMapper or the existing Fire Danger/Fire Severity pipelines,
# which are out of scope for this task.
_MS_TO_KMH = 3.6

# Verified official PROPAGATOR classes reachable from EcoGuard's exact
# Copernicus dominant_land_cover labels (backend/src/mappers/vegetation_mapper.py).
# Only semantically unambiguous labels are mapped; "Tree cover" and "Moss and
# lichen cover" have no defensible single PROPAGATOR class (see
# fire_spread_prediction.md §4.3.2 -- Task 4A "Approach A") and are
# intentionally absent from this table, as is any other/unknown label.
_DOMINANT_LAND_COVER_TO_FUEL_CLASS: dict[str, FireSpreadFuelClass] = {
    "Shrub cover": FireSpreadFuelClass.SHRUBS,
    "Grass cover": FireSpreadFuelClass.GRASSLAND,
    "Crop cover": FireSpreadFuelClass.AGRO_FORESTRY,
    "Bare cover": FireSpreadFuelClass.BARE_SOIL,
    "Built-up cover": FireSpreadFuelClass.BARE_SOIL,
    "Permanent water cover": FireSpreadFuelClass.BARE_SOIL,
    "Seasonal water cover": FireSpreadFuelClass.BARE_SOIL,
    "Snow cover": FireSpreadFuelClass.BARE_SOIL,
}


class FireSpreadInputService:
    """Gather mandatory inputs for FireSpreadCalculator from already-persisted data.

    Reuses, rather than recalculates: the active FireEvent, the latest VALID
    FireSeverityAssessment, the exact weather observations that assessment
    already traced, and its persisted vegetation snapshot. See
    backend/docs/fire_spread_prediction.md for the full sourcing.
    """

    def __init__(
        self,
        fire_event_repository: FireEventRepository | None = None,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
        weather_repository: WeatherRepository | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )
        self._weather_repository = weather_repository or WeatherRepository()

    def prepare_input(
        self,
        fire_event_id: int,
        as_of: datetime,
        horizon_minutes: int,
    ) -> FireSpreadInputResult:
        """Prepare a FireSpreadInput for an active FireEvent at a timezone-aware instant."""
        _validate_fire_event_id(fire_event_id)
        _validate_aware_datetime("as_of", as_of)
        _validate_horizon_minutes(horizon_minutes)

        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        if stored_event is None:
            return _insufficient_result(fire_event_id)
        if stored_event.event.status in _INACTIVE_EVENT_STATUSES:
            return FireSpreadInputResult(
                status=FireSpreadInputStatus.INACTIVE_EVENT,
                input_data=None,
                fire_event_id=fire_event_id,
            )
        if stored_event.event.status not in _ACTIVE_EVENT_STATUSES:
            return _insufficient_result(fire_event_id)

        latest_assessment = self._fire_severity_assessment_repository.get_latest_for_event(fire_event_id)
        if latest_assessment is None:
            return _insufficient_result(fire_event_id)
        if latest_assessment.assessment.fire_event_id != fire_event_id:
            return _insufficient_result(fire_event_id)
        if latest_assessment.assessment.status is not FireSeverityAssessmentStatus.VALID:
            return _insufficient_result(fire_event_id, severity_assessment_id=latest_assessment.assessment_id)
        if _ensure_aware(latest_assessment.assessment.assessed_at) > as_of:
            return _insufficient_result(fire_event_id, severity_assessment_id=latest_assessment.assessment_id)

        selected_weather = self._select_weather(stored_event, latest_assessment, as_of)
        if selected_weather is None:
            return _insufficient_result(fire_event_id, severity_assessment_id=latest_assessment.assessment_id)

        fuel_class = _map_dominant_land_cover(latest_assessment.assessment.vegetation_dominant_land_cover)
        if fuel_class is None:
            return _insufficient_result(
                fire_event_id,
                severity_assessment_id=latest_assessment.assessment_id,
                weather_observation_id=selected_weather.observation_id,
            )

        input_data = FireSpreadInput(
            origin_latitude=stored_event.event.latitude,
            origin_longitude=stored_event.event.longitude,
            wind_speed_kmh=selected_weather.observation.wind_speed * _MS_TO_KMH,
            wind_direction_deg=selected_weather.observation.wind_direction,
            fuel_moisture_percent=_equilibrium_moisture_percent(selected_weather.observation),
            fuel_class=fuel_class,
            horizon_minutes=horizon_minutes,
        )

        return FireSpreadInputResult(
            status=FireSpreadInputStatus.READY,
            input_data=input_data,
            fire_event_id=fire_event_id,
            severity_assessment_id=latest_assessment.assessment_id,
            weather_observation_id=selected_weather.observation_id,
        )

    def _select_weather(
        self,
        stored_event: StoredFireEvent,
        latest_assessment: StoredFireSeverityAssessment,
        as_of: datetime,
    ) -> StoredWeatherObservation | None:
        """Select one internally coherent weather observation from the
        exact observations the latest severity assessment already traced.

        Deterministic policy (fire_spread_prediction.md-aligned, per Task 5
        brief §7): keep only observations with all mandatory spread-weather
        fields, that satisfy the existing Fire Severity weather-freshness
        policy relative to `as_of`; among those, pick the station nearest to
        the FireEvent location; ties broken by (distance, newest timestamp,
        observation id).
        """
        if not latest_assessment.weather_observation_ids:
            return None

        records = self._weather_repository.get_observations_by_ids(latest_assessment.weather_observation_ids)

        eligible: list[tuple[float, float, int, StoredWeatherObservation]] = []
        for record in records:
            if not _has_required_spread_fields(record.observation):
                continue
            if not _is_fresh(record.observation.timestamp, as_of):
                continue
            distance_km = haversine_distance_km(
                stored_event.event.latitude,
                stored_event.event.longitude,
                record.station.latitude,
                record.station.longitude,
            )
            eligible.append(
                (
                    distance_km,
                    -_ensure_aware(record.observation.timestamp).timestamp(),
                    record.observation_id,
                    record,
                )
            )

        if not eligible:
            return None
        eligible.sort(key=lambda item: item[:3])
        return eligible[0][3]


def _equilibrium_moisture_percent(observation: WeatherObservation) -> float:
    """Reuse the existing FFWI equilibrium-moisture estimate as fuel moisture.

    Calls the existing, teammate-owned private helpers in
    `ffwi_calculator.py` directly rather than duplicating the formula (per
    fire_spread_prediction.md §4.5 / §15.4: the estimate is reused, the
    damping/moisture-of-extinction math is not -- that lives entirely inside
    FireSpreadCalculator, already verified in Task 4B).
    """
    temperature_f = _celsius_to_fahrenheit(observation.temperature)
    return _equilibrium_moisture_content(
        relative_humidity_pct=observation.relative_humidity,
        temperature_f=temperature_f,
    )


def _map_dominant_land_cover(dominant_land_cover: str | None) -> FireSpreadFuelClass | None:
    """Map an EcoGuard vegetation label to a verified PROPAGATOR fuel class.

    Returns None for a missing snapshot, an ambiguous label ("Tree cover",
    "Moss and lichen cover"), or any unrecognized label -- callers must treat
    None as INSUFFICIENT_DATA, never as a default/fallback fuel class.
    """
    if dominant_land_cover is None:
        return None
    return _DOMINANT_LAND_COVER_TO_FUEL_CLASS.get(dominant_land_cover)


def _has_required_spread_fields(observation: WeatherObservation) -> bool:
    return (
        observation.temperature is not None
        and observation.relative_humidity is not None
        and observation.wind_speed is not None
        and observation.wind_direction is not None
    )


def _is_fresh(timestamp: datetime, as_of: datetime) -> bool:
    observed_at = _ensure_aware(timestamp)
    age = as_of - observed_at
    return timedelta(0) <= age <= timedelta(minutes=MAX_SEVERITY_WEATHER_AGE_MINUTES)


def _insufficient_result(
    fire_event_id: int,
    severity_assessment_id: int | None = None,
    weather_observation_id: int | None = None,
) -> FireSpreadInputResult:
    return FireSpreadInputResult(
        status=FireSpreadInputStatus.INSUFFICIENT_DATA,
        input_data=None,
        fire_event_id=fire_event_id,
        severity_assessment_id=severity_assessment_id,
        weather_observation_id=weather_observation_id,
    )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _validate_fire_event_id(fire_event_id: object) -> None:
    if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
        raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")


def _validate_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")


def _validate_horizon_minutes(horizon_minutes: object) -> None:
    if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int):
        raise ValueError(f"horizon_minutes must be an integer, got {horizon_minutes!r}")
    if horizon_minutes not in SUPPORTED_HORIZON_MINUTES:
        raise ValueError(f"horizon_minutes must be one of {SUPPORTED_HORIZON_MINUTES}, got {horizon_minutes!r}")
