"""Prepare normalized inputs for the pure wildfire-spread CA calculator.

This service performs input retrieval and preparation only. It does not run
fire detection, does not recalculate fire danger or fire severity, does not
call Copernicus or IMS, does not run the CA propagation, and does not
persist a FireSpreadPrediction. See backend/docs/fire_spread_prediction.md
for the reuse-first architectural rule this service implements.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.calculators.fire_danger.ffwi_calculator import _celsius_to_fahrenheit, _equilibrium_moisture_content
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_input_result import FireSpreadInputResult
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason
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

# WeatherObservation.wind_speed is canonically km/h (IMS m/s is converted once,
# at ingestion, by WeatherMapper), which is exactly FireSpreadCalculator's
# `wind_speed_kmh` contract - so it is passed through with no conversion here.

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


@dataclass(frozen=True)
class FireSpreadSharedContext:
    """Every FireSpreadInput-determining value that does NOT depend on
    horizon_minutes (performance pass: profiling showed prepare_input() was
    called once per horizon - 2x per refresh - independently re-loading the
    identical FireEvent/severity/weather each time, a pure N+1). Loaded once
    per FireEvent+as_of by prepare_shared_context() and reused across every
    horizon via build_input_for_horizon(). Opaque to callers - only
    FireSpreadInputService itself interprets these fields."""

    status: FireSpreadInputStatus
    fire_event_id: int
    origin_latitude: float | None = None
    origin_longitude: float | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    fuel_moisture_percent: float | None = None
    fuel_class: FireSpreadFuelClass | None = None
    severity_assessment_id: int | None = None
    weather_observation_id: int | None = None
    insufficient_data_reason: FireSpreadInsufficientDataReason | None = None


class FireSpreadInputService:
    """Gather mandatory inputs for FireSpreadCalculator from already-persisted data.

    Reuses, rather than recalculates: the active FireEvent, the latest VALID
    FireSeverityAssessment available at or before the requested `as_of`, the
    exact weather observations that assessment already traced, and its
    persisted vegetation snapshot. See backend/docs/fire_spread_prediction.md
    for the full sourcing.
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
        """Prepare a FireSpreadInput for an active FireEvent at a timezone-aware instant.

        Unchanged behavior/signature for single-horizon callers - internally
        now just composes prepare_shared_context() + build_input_for_horizon()
        (performance pass: see FireSpreadSharedContext's docstring). Validates
        horizon_minutes before any I/O, exactly as before.
        """
        _validate_horizon_minutes(horizon_minutes)
        context = self.prepare_shared_context(fire_event_id, as_of)
        return self.build_input_for_horizon(context, horizon_minutes)

    def prepare_shared_context(self, fire_event_id: int, as_of: datetime) -> FireSpreadSharedContext:
        """Load every horizon-independent input ONCE - reused across every
        horizon by a multi-horizon caller (FireSpreadRefreshOrchestrator) to
        avoid re-fetching the identical FireEvent/severity/weather once per
        horizon. Fetches the FireEvent by id and reads the latest severity
        assessment from persistence - unchanged behavior/signature for
        existing callers. See prepare_shared_context_for_event() for the
        performance-pass overload used by OperationalRefreshOrchestrator.
        """
        _validate_fire_event_id(fire_event_id)
        _validate_aware_datetime("as_of", as_of)

        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        if stored_event is None:
            return FireSpreadSharedContext(
                status=FireSpreadInputStatus.INSUFFICIENT_DATA,
                fire_event_id=fire_event_id,
                insufficient_data_reason=FireSpreadInsufficientDataReason.EVENT_UNAVAILABLE,
            )
        return self.prepare_shared_context_for_event(stored_event, as_of)

    def prepare_shared_context_for_event(
        self,
        stored_event: StoredFireEvent,
        as_of: datetime,
        *,
        resolved_severity: StoredFireSeverityAssessment | None = None,
    ) -> FireSpreadSharedContext:
        """Same preparation as prepare_shared_context(), but for an
        ALREADY-LOADED StoredFireEvent, and optionally an ALREADY-RESOLVED
        severity assessment (performance pass: avoids a redundant FireEvent
        fetch, and - when `resolved_severity` is supplied - a redundant
        "latest severity" query, since OperationalRefreshOrchestrator has
        just established the authoritative severity result for this exact
        refresh cycle, whether newly computed or reused).

        `resolved_severity=None` (the default) falls back to reading the
        latest persisted severity from the repository, exactly like
        prepare_shared_context() - used when the caller has no fresher
        in-cycle result (e.g. severity was not re-evaluated this trigger)
        or is an external/legacy caller.
        """
        _validate_aware_datetime("as_of", as_of)
        fire_event_id = stored_event.id
        if stored_event.event.status in _INACTIVE_EVENT_STATUSES:
            return FireSpreadSharedContext(status=FireSpreadInputStatus.INACTIVE_EVENT, fire_event_id=fire_event_id)
        if stored_event.event.status not in _ACTIVE_EVENT_STATUSES:
            return FireSpreadSharedContext(
                status=FireSpreadInputStatus.INSUFFICIENT_DATA,
                fire_event_id=fire_event_id,
                insufficient_data_reason=FireSpreadInsufficientDataReason.EVENT_UNAVAILABLE,
            )

        latest_assessment = (
            resolved_severity
            if resolved_severity is not None
            else self._fire_severity_assessment_repository.get_latest_for_event_as_of(fire_event_id, as_of)
        )
        if latest_assessment is None or latest_assessment.assessment.fire_event_id != fire_event_id:
            return FireSpreadSharedContext(
                status=FireSpreadInputStatus.INSUFFICIENT_DATA,
                fire_event_id=fire_event_id,
                insufficient_data_reason=FireSpreadInsufficientDataReason.MISSING_SEVERITY,
            )
        if latest_assessment.assessment.status is not FireSeverityAssessmentStatus.VALID:
            return FireSpreadSharedContext(
                status=FireSpreadInputStatus.INSUFFICIENT_DATA,
                fire_event_id=fire_event_id,
                severity_assessment_id=latest_assessment.assessment_id,
                insufficient_data_reason=FireSpreadInsufficientDataReason.SEVERITY_NOT_VALID,
            )

        selected_weather, weather_reason = self._select_weather(stored_event, latest_assessment, as_of)
        if selected_weather is None:
            return FireSpreadSharedContext(
                status=FireSpreadInputStatus.INSUFFICIENT_DATA,
                fire_event_id=fire_event_id,
                severity_assessment_id=latest_assessment.assessment_id,
                insufficient_data_reason=weather_reason,
            )

        dominant_land_cover = latest_assessment.assessment.vegetation_dominant_land_cover
        fuel_class = _map_dominant_land_cover(dominant_land_cover)
        if fuel_class is None:
            return FireSpreadSharedContext(
                status=FireSpreadInputStatus.INSUFFICIENT_DATA,
                fire_event_id=fire_event_id,
                severity_assessment_id=latest_assessment.assessment_id,
                weather_observation_id=selected_weather.observation_id,
                insufficient_data_reason=(
                    FireSpreadInsufficientDataReason.MISSING_VEGETATION
                    if dominant_land_cover is None
                    else FireSpreadInsufficientDataReason.UNSUPPORTED_VEGETATION
                ),
            )

        return FireSpreadSharedContext(
            status=FireSpreadInputStatus.READY,
            fire_event_id=fire_event_id,
            origin_latitude=stored_event.event.latitude,
            origin_longitude=stored_event.event.longitude,
            wind_speed_kmh=selected_weather.observation.wind_speed,
            wind_direction_deg=selected_weather.observation.wind_direction,
            fuel_moisture_percent=_equilibrium_moisture_percent(selected_weather.observation),
            fuel_class=fuel_class,
            severity_assessment_id=latest_assessment.assessment_id,
            weather_observation_id=selected_weather.observation_id,
        )

    @staticmethod
    def build_input_for_horizon(context: FireSpreadSharedContext, horizon_minutes: int) -> FireSpreadInputResult:
        """Pure, no I/O: build the horizon-specific FireSpreadInputResult from
        an already-loaded FireSpreadSharedContext."""
        _validate_horizon_minutes(horizon_minutes)
        if context.status is not FireSpreadInputStatus.READY:
            return FireSpreadInputResult(
                status=context.status,
                input_data=None,
                fire_event_id=context.fire_event_id,
                severity_assessment_id=context.severity_assessment_id,
                weather_observation_id=context.weather_observation_id,
                insufficient_data_reason=context.insufficient_data_reason,
            )

        input_data = FireSpreadInput(
            origin_latitude=context.origin_latitude,
            origin_longitude=context.origin_longitude,
            wind_speed_kmh=context.wind_speed_kmh,
            wind_direction_deg=context.wind_direction_deg,
            fuel_moisture_percent=context.fuel_moisture_percent,
            fuel_class=context.fuel_class,
            horizon_minutes=horizon_minutes,
        )
        return FireSpreadInputResult(
            status=FireSpreadInputStatus.READY,
            input_data=input_data,
            fire_event_id=context.fire_event_id,
            severity_assessment_id=context.severity_assessment_id,
            weather_observation_id=context.weather_observation_id,
        )

    def _select_weather(
        self,
        stored_event: StoredFireEvent,
        latest_assessment: StoredFireSeverityAssessment,
        as_of: datetime,
    ) -> tuple[StoredWeatherObservation | None, FireSpreadInsufficientDataReason | None]:
        """Select one internally coherent weather observation from the
        exact observations the latest severity assessment already traced.

        Deterministic policy (fire_spread_prediction.md-aligned, per Task 5
        brief §7): keep only observations with all mandatory spread-weather
        fields, that satisfy the existing Fire Severity weather-freshness
        policy relative to `as_of`; among those, pick the station nearest to
        the FireEvent location; ties broken by (distance, newest timestamp,
        observation id).

        Returns (observation, None), or (None, reason) when nothing is
        eligible, with precedence: no retrievable observation ->
        MISSING_WEATHER; none has all required fields -> INCOMPLETE_WEATHER;
        any complete observation older than the freshness window ->
        STALE_WEATHER; otherwise (all complete ones in the future) ->
        FUTURE_WEATHER.
        """
        if not latest_assessment.weather_observation_ids:
            return None, FireSpreadInsufficientDataReason.MISSING_WEATHER

        records = self._weather_repository.get_observations_by_ids(latest_assessment.weather_observation_ids)
        if not records:
            return None, FireSpreadInsufficientDataReason.MISSING_WEATHER

        has_complete = False
        has_stale = False
        eligible: list[tuple[float, float, int, StoredWeatherObservation]] = []
        for record in records:
            if not _has_required_spread_fields(record.observation):
                continue
            has_complete = True
            if not _is_fresh(record.observation.timestamp, as_of):
                if _ensure_aware(record.observation.timestamp) <= as_of:
                    has_stale = True
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
            if not has_complete:
                return None, FireSpreadInsufficientDataReason.INCOMPLETE_WEATHER
            if has_stale:
                return None, FireSpreadInsufficientDataReason.STALE_WEATHER
            return None, FireSpreadInsufficientDataReason.FUTURE_WEATHER
        eligible.sort(key=lambda item: item[:3])
        return eligible[0][3], None


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
