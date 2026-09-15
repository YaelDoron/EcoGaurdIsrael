"""Tests for FireSeverityInputService."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.external.copernicus import (
    CopernicusCoverFraction,
    CopernicusLandCoverStatistics,
    CopernicusServiceUnavailableError,
)
from src.mappers.vegetation_mapper import VegetationMapper
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityInputStatus,
    SatelliteHotspot,
    VegetationData,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.satellite_hotspot_repository import StoredSatelliteHotspot
from src.repositories.weather_repository import StoredWeatherObservation
from src.services.fire_severity.fire_severity_input_config import (
    MAX_SEVERITY_SATELLITE_AGE_HOURS,
    MAX_SEVERITY_WEATHER_AGE_MINUTES,
    SEVERITY_WEATHER_RADIUS_KM,
    VEGETATION_RADIUS_KM,
)
from src.services.fire_severity.fire_severity_input_service import FireSeverityInputService

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


class FakeFireEventRepository:
    def __init__(self, stored_event: StoredFireEvent | None) -> None:
        self.stored_event = stored_event
        self.calls = []

    def get_by_id(self, fire_event_id: int):
        self.calls.append(fire_event_id)
        return self.stored_event


class FakeWeatherRepository:
    def __init__(self, records: list[StoredWeatherObservation]) -> None:
        self.records = records
        self.calls = []

    def get_recent_observations_for_area_candidates(
        self,
        latitude,
        longitude,
        radius_km,
        start_time,
        end_time,
    ):
        self.calls.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        return self.records


class FakeSatelliteRepository:
    def __init__(self, records: dict[int, StoredSatelliteHotspot]) -> None:
        self.records = records
        self.calls = []

    def get_by_id(self, hotspot_id: int):
        self.calls.append(hotspot_id)
        return self.records.get(hotspot_id)


class FakeLandCoverProvider:
    def __init__(self, statistics=None, error: Exception | None = None) -> None:
        self.statistics = statistics
        self.error = error
        self.calls = []

    def get_land_cover_statistics(self, latitude, longitude, radius_km):
        self.calls.append({"latitude": latitude, "longitude": longitude, "radius_km": radius_km})
        if self.error is not None:
            raise self.error
        return self.statistics


def make_event(status=FireEventStatus.SUSPECTED, evidence_refs=None) -> StoredFireEvent:
    return StoredFireEvent(
        id=10,
        event=FireEvent(
            latitude=32.731,
            longitude=35.046,
            detected_at=AS_OF - timedelta(minutes=20),
            updated_at=AS_OF - timedelta(minutes=5),
            status=status,
            detection_confidence=0.6 if status is FireEventStatus.SUSPECTED else 0.85,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
        supporting_evidence=tuple(evidence_refs or (FireEvidenceRef(FireEvidenceType.SATELLITE, 1),)),
    )


def make_weather(
    station_id=1,
    observation_id=101,
    minutes_old=5,
    latitude=32.731,
    longitude=35.046,
    wind_speed=20.0,
    relative_humidity=30.0,
) -> StoredWeatherObservation:
    external_station_id = 900000 + station_id
    return StoredWeatherObservation(
        observation_id=observation_id,
        station_id=station_id,
        station=WeatherStation(
            external_station_id=external_station_id,
            name=f"Station {station_id}",
            latitude=latitude,
            longitude=longitude,
        ),
        observation=WeatherObservation(
            station_external_id=external_station_id,
            timestamp=AS_OF - timedelta(minutes=minutes_old),
            temperature=None,
            relative_humidity=relative_humidity,
            wind_speed=wind_speed,
        ),
    )


def make_hotspot(hotspot_id=1, hours_old=1, frp=50.0) -> StoredSatelliteHotspot:
    return StoredSatelliteHotspot(
        id=hotspot_id,
        hotspot=SatelliteHotspot(
            latitude=32.731,
            longitude=35.046,
            detected_at=AS_OF - timedelta(hours=hours_old),
            confidence="h",
            frp=frp,
        ),
    )


def build_service(
    *,
    stored_event=None,
    weather_records=None,
    satellite_records=None,
    statistics=None,
    copernicus_error: Exception | None = None,
):
    event_repo = FakeFireEventRepository(stored_event if stored_event is not None else make_event())
    weather_repo = FakeWeatherRepository(weather_records if weather_records is not None else [make_weather()])
    satellite_repo = FakeSatelliteRepository(satellite_records if satellite_records is not None else {1: make_hotspot()})
    provider = FakeLandCoverProvider(statistics=statistics, error=copernicus_error)
    service = FireSeverityInputService(
        fire_event_repository=event_repo,
        weather_repository=weather_repo,
        satellite_hotspot_repository=satellite_repo,
        land_cover_client=provider,
        vegetation_mapper=VegetationMapper(),
    )
    return service, event_repo, weather_repo, satellite_repo, provider


@pytest.mark.parametrize("status", [FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED])
def test_active_fire_event_may_prepare_input(status):
    service, *_ = build_service(stored_event=make_event(status=status))

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.READY
    assert result.input_data.frp_mw == pytest.approx(50)


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_fire_event_returns_inactive_without_loading_inputs(status):
    service, _, weather_repo, satellite_repo, provider = build_service(stored_event=make_event(status=status))

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.INACTIVE_EVENT
    assert result.input_data is None
    assert weather_repo.calls == []
    assert satellite_repo.calls == []
    assert provider.calls == []


def test_valid_recent_weather_and_frp_return_ready():
    service, _, weather_repo, _, _ = build_service()

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.READY
    assert result.input_data.wind_speed_kmh == pytest.approx(20)
    assert result.input_data.relative_humidity_pct == pytest.approx(30)
    assert result.weather_observation_ids == (101,)
    assert weather_repo.calls[0]["radius_km"] == SEVERITY_WEATHER_RADIUS_KM
    assert weather_repo.calls[0]["start_time"] == AS_OF - timedelta(minutes=MAX_SEVERITY_WEATHER_AGE_MINUTES)
    assert weather_repo.calls[0]["end_time"] == AS_OF


def test_weather_exactly_thirty_minutes_old_is_valid():
    service, *_ = build_service(weather_records=[make_weather(minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES)])

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.READY


def test_weather_older_than_thirty_minutes_is_excluded():
    service, *_ = build_service(weather_records=[make_weather(minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES + 0.1)])

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.INSUFFICIENT_DATA
    assert result.input_data is None


def test_future_weather_is_excluded():
    service, *_ = build_service(weather_records=[make_weather(minutes_old=-1)])

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.INSUFFICIENT_DATA


@pytest.mark.parametrize("overrides", [{"wind_speed": None}, {"relative_humidity": None}])
def test_missing_required_weather_fields_are_insufficient(overrides):
    service, *_ = build_service(weather_records=[make_weather(**overrides)])

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.INSUFFICIENT_DATA


def test_multiple_valid_weather_stations_are_averaged_and_ids_preserved():
    service, *_ = build_service(
        weather_records=[
            make_weather(station_id=2, observation_id=202, wind_speed=30, relative_humidity=40),
            make_weather(station_id=1, observation_id=101, wind_speed=10, relative_humidity=20),
        ]
    )

    result = service.prepare_input(10, AS_OF)

    assert result.input_data.wind_speed_kmh == pytest.approx(20)
    assert result.input_data.relative_humidity_pct == pytest.approx(30)
    assert result.weather_observation_ids == (101, 202)


def test_one_recent_associated_hotspot_is_selected():
    service, *_ = build_service()

    result = service.prepare_input(10, AS_OF)

    assert result.satellite_hotspot_ids == (1,)
    assert result.selected_frp_hotspot_id == 1
    assert result.input_data.frp_mw == pytest.approx(50)


def test_several_hotspots_select_max_frp_and_preserve_selected_id():
    event = make_event(
        evidence_refs=(
            FireEvidenceRef(FireEvidenceType.SATELLITE, 1),
            FireEvidenceRef(FireEvidenceType.SATELLITE, 2),
            FireEvidenceRef(FireEvidenceType.SATELLITE, 3),
        )
    )
    service, *_ = build_service(
        stored_event=event,
        satellite_records={
            1: make_hotspot(1, frp=35),
            2: make_hotspot(2, frp=72),
            3: make_hotspot(3, frp=55),
        },
    )

    result = service.prepare_input(10, AS_OF)

    assert result.satellite_hotspot_ids == (1, 2, 3)
    assert result.selected_frp_hotspot_id == 2
    assert result.input_data.frp_mw == pytest.approx(72)


def test_hotspot_exactly_six_hours_old_is_valid():
    service, *_ = build_service(satellite_records={1: make_hotspot(hours_old=MAX_SEVERITY_SATELLITE_AGE_HOURS)})

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.READY


def test_hotspot_older_than_six_hours_is_excluded():
    service, *_ = build_service(satellite_records={1: make_hotspot(hours_old=MAX_SEVERITY_SATELLITE_AGE_HOURS + 0.1)})

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.INSUFFICIENT_DATA


def test_future_hotspot_is_excluded():
    service, *_ = build_service(satellite_records={1: make_hotspot(hours_old=-1)})

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.INSUFFICIENT_DATA


def test_unrelated_hotspot_not_attached_to_event_is_not_used():
    service, _, _, satellite_repo, _ = build_service(
        satellite_records={
            1: make_hotspot(1, frp=40),
            2: make_hotspot(2, frp=90),
        }
    )

    result = service.prepare_input(10, AS_OF)

    assert satellite_repo.calls == [1]
    assert result.selected_frp_hotspot_id == 1
    assert result.input_data.frp_mw == pytest.approx(40)


@pytest.mark.parametrize(
    "satellite_records",
    [
        {},
        {1: make_hotspot(1, frp=None)},
    ],
)
def test_no_usable_associated_frp_is_insufficient(satellite_records):
    service, *_ = build_service(satellite_records=satellite_records)

    assert service.prepare_input(10, AS_OF).status is FireSeverityInputStatus.INSUFFICIENT_DATA


def test_valid_vegetation_returns_ready_with_fuel_score_and_traceability():
    service, _, _, _, provider = build_service(
        statistics=CopernicusLandCoverStatistics(
            cover_fractions=(CopernicusCoverFraction("tree", 1.0),),
            source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
            dataset_year=2019,
            radius_km=VEGETATION_RADIUS_KM,
        )
    )

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.READY
    assert result.input_data.vegetation_fuel_score == pytest.approx(0.9)
    assert result.vegetation_data.dominant_land_cover == "Tree cover"
    assert provider.calls == [{"latitude": 32.731, "longitude": 35.046, "radius_km": VEGETATION_RADIUS_KM}]


@pytest.mark.parametrize(
    "statistics",
    [
        None,
        CopernicusLandCoverStatistics(
            cover_fractions=(CopernicusCoverFraction("unknown", 1.0),),
            source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
            dataset_year=2019,
            radius_km=VEGETATION_RADIUS_KM,
        ),
    ],
)
def test_missing_or_unmapped_vegetation_is_ready_when_mandatory_inputs_exist(statistics):
    service, *_ = build_service(statistics=statistics)

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.READY
    assert result.input_data.vegetation_fuel_score is None
    assert result.vegetation_data is None


def test_copernicus_failure_is_ready_without_vegetation_when_mandatory_inputs_exist():
    service, *_ = build_service(
        copernicus_error=CopernicusServiceUnavailableError("Copernicus statistics request timed out.")
    )

    result = service.prepare_input(10, AS_OF)

    assert result.status is FireSeverityInputStatus.READY
    assert result.input_data.vegetation_fuel_score is None
    assert result.vegetation_data is None


def test_same_inputs_are_deterministic():
    service, *_ = build_service(
        statistics=CopernicusLandCoverStatistics(
            cover_fractions=(CopernicusCoverFraction("grass", 1.0),),
            source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
            dataset_year=2019,
            radius_km=VEGETATION_RADIUS_KM,
        )
    )

    first = service.prepare_input(10, AS_OF)
    second = service.prepare_input(10, AS_OF)

    assert first == second


def test_prepare_input_requires_timezone_aware_as_of():
    service, *_ = build_service()

    with pytest.raises(ValueError):
        service.prepare_input(10, datetime(2026, 9, 14, 12, 0))
