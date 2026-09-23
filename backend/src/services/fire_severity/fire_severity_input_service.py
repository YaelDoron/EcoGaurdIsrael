"""Prepare normalized inputs for active wildfire severity calculation."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import math

from src.external.copernicus import CopernicusClientError, CopernicusLandCoverClient
from src.mappers.vegetation_mapper import VegetationMapper
from src.models import (
    FireEventStatus,
    FireEvidenceType,
    FireSeverityInput,
    FireSeverityInputResult,
    FireSeverityInputStatus,
)
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository, StoredSatelliteHotspot
from src.repositories.weather_repository import StoredWeatherObservation, WeatherRepository
from src.services.fire_severity.fire_severity_input_config import (
    MAX_SEVERITY_SATELLITE_AGE_HOURS,
    MAX_SEVERITY_WEATHER_AGE_MINUTES,
    SEVERITY_WEATHER_RADIUS_KM,
    VEGETATION_RADIUS_KM,
)

_ACTIVE_EVENT_STATUSES = {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}
_INACTIVE_EVENT_STATUSES = {FireEventStatus.RESOLVED, FireEventStatus.DISMISSED}
_EARTH_RADIUS_KM = 6371.0088


class FireSeverityInputService:
    """Gather mandatory and optional inputs for FireSeverityCalculator.

    This service performs input retrieval and preparation only. It does not
    calculate severity scores, classify severity levels, persist assessments,
    recalculate fire detection, or inspect scenario metadata.
    """

    def __init__(
        self,
        fire_event_repository: FireEventRepository | None = None,
        weather_repository: WeatherRepository | None = None,
        satellite_hotspot_repository: SatelliteHotspotRepository | None = None,
        land_cover_client: CopernicusLandCoverClient | None = None,
        vegetation_mapper: VegetationMapper | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._weather_repository = weather_repository or WeatherRepository()
        self._satellite_hotspot_repository = satellite_hotspot_repository or SatelliteHotspotRepository()
        self._land_cover_client = land_cover_client or CopernicusLandCoverClient()
        self._vegetation_mapper = vegetation_mapper or VegetationMapper()

    def prepare_input(self, fire_event_id: int, as_of: datetime) -> FireSeverityInputResult:
        """Prepare FireSeverityInput for an active FireEvent at a timezone-aware instant.

        Fetches the FireEvent by id, then delegates to prepare_input_for_event()
        - unchanged behavior/signature for existing callers (API routers, the
        legacy simulation coordinator).
        """
        _validate_fire_event_id(fire_event_id)
        _validate_aware_datetime("as_of", as_of)

        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        if stored_event is None:
            return _insufficient_result(fire_event_id)
        return self.prepare_input_for_event(stored_event, as_of)

    def prepare_input_for_event(self, stored_event: StoredFireEvent, as_of: datetime) -> FireSeverityInputResult:
        """Same preparation as prepare_input(), but for an ALREADY-LOADED
        StoredFireEvent - performance pass: avoids a redundant FireEvent
        fetch when the caller (OperationalRefreshOrchestrator) already has
        one for this refresh cycle. Never touches FireEventRepository.
        """
        _validate_aware_datetime("as_of", as_of)
        fire_event_id = stored_event.id
        if stored_event.event.status in _INACTIVE_EVENT_STATUSES:
            return FireSeverityInputResult(
                status=FireSeverityInputStatus.INACTIVE_EVENT,
                input_data=None,
                fire_event_id=fire_event_id,
            )
        if stored_event.event.status not in _ACTIVE_EVENT_STATUSES:
            return _insufficient_result(fire_event_id)

        weather_records = self._select_weather(stored_event, as_of)
        satellite_records = self._select_satellite_hotspots(stored_event, as_of)
        selected_frp_hotspot = _select_max_frp_hotspot(satellite_records)

        vegetation_data = self._load_vegetation(stored_event)

        if not weather_records or selected_frp_hotspot is None:
            return FireSeverityInputResult(
                status=FireSeverityInputStatus.INSUFFICIENT_DATA,
                input_data=None,
                fire_event_id=fire_event_id,
                weather_observation_ids=tuple(record.observation_id for record in weather_records),
                satellite_hotspot_ids=tuple(record.id for record in satellite_records),
                selected_frp_hotspot_id=selected_frp_hotspot.id if selected_frp_hotspot is not None else None,
                vegetation_data=vegetation_data,
            )

        wind_speed_kmh = sum(record.observation.wind_speed for record in weather_records) / len(weather_records)
        relative_humidity_pct = sum(record.observation.relative_humidity for record in weather_records) / len(
            weather_records
        )

        return FireSeverityInputResult(
            status=FireSeverityInputStatus.READY,
            input_data=FireSeverityInput(
                frp_mw=selected_frp_hotspot.hotspot.frp,
                wind_speed_kmh=wind_speed_kmh,
                relative_humidity_pct=relative_humidity_pct,
                vegetation_fuel_score=vegetation_data.fuel_score if vegetation_data is not None else None,
            ),
            fire_event_id=fire_event_id,
            weather_observation_ids=tuple(record.observation_id for record in weather_records),
            satellite_hotspot_ids=tuple(record.id for record in satellite_records),
            selected_frp_hotspot_id=selected_frp_hotspot.id,
            vegetation_data=vegetation_data,
        )

    def _select_weather(
        self,
        stored_event: StoredFireEvent,
        as_of: datetime,
    ) -> tuple[StoredWeatherObservation, ...]:
        earliest = as_of - timedelta(minutes=MAX_SEVERITY_WEATHER_AGE_MINUTES)
        candidates = self._weather_repository.get_recent_observations_for_area_candidates(
            latitude=stored_event.event.latitude,
            longitude=stored_event.event.longitude,
            radius_km=SEVERITY_WEATHER_RADIUS_KM,
            start_time=earliest,
            end_time=as_of,
        )
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
            if not _station_is_within_radius(stored_event, record):
                continue
            if not _weather_observation_is_fresh(record.observation.timestamp, as_of):
                continue
            if not _weather_observation_has_required_fields(record):
                continue
            selected_by_station[record.station_id] = record
        return tuple(selected_by_station.values())

    def _select_satellite_hotspots(
        self,
        stored_event: StoredFireEvent,
        as_of: datetime,
    ) -> tuple[StoredSatelliteHotspot, ...]:
        satellite_ids = tuple(
            ref.evidence_id
            for ref in stored_event.supporting_evidence
            if ref.evidence_type is FireEvidenceType.SATELLITE
        )
        # Performance pass: batched in one query (get_by_ids) instead of one
        # get_by_id() round trip per evidence reference - profiling showed
        # severity's input loading dominated its total cost, and this loop
        # was a pure N+1 over what is often 1-2 satellite references.
        by_id = {record.id: record for record in self._satellite_hotspot_repository.get_by_ids(satellite_ids)}
        records = []
        for hotspot_id in sorted(satellite_ids):
            record = by_id.get(hotspot_id)
            if record is None:
                continue
            if not _satellite_hotspot_is_recent(record, as_of):
                continue
            if record.hotspot.frp is None:
                continue
            records.append(record)
        return tuple(records)

    def _load_vegetation(self, stored_event: StoredFireEvent):
        try:
            statistics = self._land_cover_client.get_land_cover_statistics(
                latitude=stored_event.event.latitude,
                longitude=stored_event.event.longitude,
                radius_km=VEGETATION_RADIUS_KM,
            )
        except CopernicusClientError:
            return None
        return self._vegetation_mapper.map_statistics(statistics)


def _select_max_frp_hotspot(
    records: tuple[StoredSatelliteHotspot, ...],
) -> StoredSatelliteHotspot | None:
    if not records:
        return None
    return sorted(records, key=lambda record: (-record.hotspot.frp, record.id))[0]


def _insufficient_result(fire_event_id: int) -> FireSeverityInputResult:
    return FireSeverityInputResult(
        status=FireSeverityInputStatus.INSUFFICIENT_DATA,
        input_data=None,
        fire_event_id=fire_event_id,
    )


def _station_is_within_radius(stored_event: StoredFireEvent, record: StoredWeatherObservation) -> bool:
    distance = haversine_distance_km(
        stored_event.event.latitude,
        stored_event.event.longitude,
        record.station.latitude,
        record.station.longitude,
    )
    return distance <= SEVERITY_WEATHER_RADIUS_KM


def _weather_observation_is_fresh(timestamp: datetime, as_of: datetime) -> bool:
    observed_at = _ensure_aware(timestamp)
    age = as_of - observed_at
    return timedelta(0) <= age <= timedelta(minutes=MAX_SEVERITY_WEATHER_AGE_MINUTES)


def _weather_observation_has_required_fields(record: StoredWeatherObservation) -> bool:
    return record.observation.relative_humidity is not None and record.observation.wind_speed is not None


def _satellite_hotspot_is_recent(record: StoredSatelliteHotspot, as_of: datetime) -> bool:
    detected_at = _ensure_aware(record.hotspot.detected_at)
    age = as_of - detected_at
    return timedelta(0) <= age <= timedelta(hours=MAX_SEVERITY_SATELLITE_AGE_HOURS)


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
    return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _timestamp_sort_value(value: datetime) -> float:
    return value.timestamp()


def _validate_fire_event_id(fire_event_id: object) -> None:
    if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
        raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")


def _validate_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
