"""Tests for GlobalIncidentDemandBuilder (Stage 5 of the Global
Multi-Incident Optimizer refactor, Task 5), against a real SQLite-backed
FireSeverityAssessmentRepository - mirroring the same VALID/
INSUFFICIENT_DATA/INACTIVE_EVENT/missing-entirely scenarios already
covered for the repository itself."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    SatelliteHotspot,
)
from src.models.demand_source import DemandSource
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.global_planning.global_incident_demand_builder import GlobalIncidentDemandBuilder

ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EVENT_TIME = ASSESSED_AT - timedelta(minutes=20)


@pytest.fixture
def severity_repository(sqlite_session_factory) -> FireSeverityAssessmentRepository:
    return FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def fire_event_repository(sqlite_session_factory) -> FireEventRepository:
    return FireEventRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def satellite_repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def weather_repository(sqlite_session_factory) -> WeatherRepository:
    return WeatherRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def builder(severity_repository) -> GlobalIncidentDemandBuilder:
    return GlobalIncidentDemandBuilder(fire_severity_assessment_repository=severity_repository)


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=EVENT_TIME,
        updated_at=ASSESSED_AT - timedelta(minutes=5),
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def make_assessment(fire_event_id: int, **overrides) -> FireSeverityAssessment:
    defaults = dict(
        fire_event_id=fire_event_id,
        assessed_at=ASSESSED_AT,
        status=FireSeverityAssessmentStatus.VALID,
        score=82.5,
        level=FireSeverityLevel.CRITICAL,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
        vegetation_dataset_year=2019,
        vegetation_radius_km=1.0,
        vegetation_dominant_land_cover="Tree cover",
        vegetation_fuel_score=0.9,
    )
    defaults.update(overrides)
    return FireSeverityAssessment(**defaults)


def persist_hotspot(satellite_repository: SatelliteHotspotRepository) -> int:
    hotspot = SatelliteHotspot(
        latitude=32.731, longitude=35.046, detected_at=EVENT_TIME, confidence="h", frp=72.0, satellite="N20",
    )
    satellite_repository.save_hotspot(hotspot)
    return satellite_repository.get_recent_hotspots(as_of=ASSESSED_AT, lookback_minutes=360)[0].id


def persist_fire_event(fire_event_repository: FireEventRepository, satellite_repository: SatelliteHotspotRepository) -> tuple[int, int]:
    hotspot_id = persist_hotspot(satellite_repository)
    saved = fire_event_repository.create_event(make_event(), (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),))
    return saved.id, hotspot_id


def persist_weather(weather_repository: WeatherRepository, station_offset: int = 0) -> int:
    station = WeatherStation(
        external_station_id=920000 + station_offset,
        name=f"Station {station_offset}",
        latitude=32.731,
        longitude=35.046,
    )
    weather_repository.save_station(station)
    observation = WeatherObservation(
        station_external_id=station.external_station_id,
        timestamp=ASSESSED_AT - timedelta(minutes=5 + station_offset),
        temperature=30.0,
        relative_humidity=25.0,
        wind_speed=20.0,
    )
    weather_repository.save_observation(observation)
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.731, longitude=35.046, radius_km=5.0,
        start_time=ASSESSED_AT - timedelta(minutes=30), end_time=ASSESSED_AT,
    )
    return next(
        record.observation_id
        for record in candidates
        if record.observation.station_external_id == station.external_station_id
    )


def test_valid_assessment_uses_policy_demand_for_its_severity_level(
    builder, severity_repository, fire_event_repository, satellite_repository, weather_repository
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    saved = severity_repository.save_assessment(
        make_assessment(fire_event_id, level=FireSeverityLevel.HIGH, score=60.0),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    (demand,) = builder.build((fire_event_id,), ASSESSED_AT)

    assert demand.demand_source is DemandSource.SEVERITY_ASSESSMENT
    assert demand.severity_assessment_id == saved.assessment_id
    assert demand.severity_level is FireSeverityLevel.HIGH
    assert demand.severity_score == pytest.approx(60.0)
    assert (demand.minimum_resources, demand.desired_resources) == (2, 3)


def test_insufficient_data_status_falls_back_to_policy_fallback(
    builder, severity_repository, fire_event_repository, satellite_repository
):
    fire_event_id, _hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    severity_repository.save_assessment(
        make_assessment(fire_event_id, status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None),
        weather_observation_ids=(),
        satellite_hotspot_ids=(),
        selected_frp_hotspot_id=None,
    )

    (demand,) = builder.build((fire_event_id,), ASSESSED_AT)

    assert demand.demand_source is DemandSource.INSUFFICIENT_SEVERITY
    assert demand.severity_level is None
    assert demand.severity_assessment_id is None
    assert (demand.minimum_resources, demand.desired_resources) == (1, 1)


def test_inactive_event_status_falls_back_to_policy_fallback(
    builder, severity_repository, fire_event_repository, satellite_repository
):
    fire_event_id, _hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    severity_repository.save_assessment(
        make_assessment(fire_event_id, status=FireSeverityAssessmentStatus.INACTIVE_EVENT, score=None, level=None),
        weather_observation_ids=(),
        satellite_hotspot_ids=(),
        selected_frp_hotspot_id=None,
    )

    (demand,) = builder.build((fire_event_id,), ASSESSED_AT)

    assert demand.demand_source is DemandSource.INSUFFICIENT_SEVERITY
    assert (demand.minimum_resources, demand.desired_resources) == (1, 1)


def test_missing_assessment_entirely_falls_back_to_policy_fallback(
    builder, fire_event_repository, satellite_repository
):
    fire_event_id, _hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)

    (demand,) = builder.build((fire_event_id,), ASSESSED_AT)

    assert demand.demand_source is DemandSource.INSUFFICIENT_SEVERITY
    assert demand.severity_level is None
    assert (demand.minimum_resources, demand.desired_resources) == (1, 1)


def test_future_assessment_is_ignored_and_falls_back_as_of_the_given_instant(
    builder, severity_repository, fire_event_repository, satellite_repository, weather_repository
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    severity_repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT + timedelta(minutes=5), level=FireSeverityLevel.CRITICAL, score=90.0),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    (demand,) = builder.build((fire_event_id,), ASSESSED_AT)

    assert demand.demand_source is DemandSource.INSUFFICIENT_SEVERITY


def test_builds_one_demand_per_event_independently(
    builder, severity_repository, fire_event_repository, satellite_repository, weather_repository
):
    critical_event_id, critical_hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    critical_weather_id = persist_weather(weather_repository, station_offset=1)
    severity_repository.save_assessment(
        make_assessment(critical_event_id, level=FireSeverityLevel.CRITICAL, score=95.0),
        weather_observation_ids=(critical_weather_id,),
        satellite_hotspot_ids=(critical_hotspot_id,),
        selected_frp_hotspot_id=critical_hotspot_id,
    )
    low_event_id, low_hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    low_weather_id = persist_weather(weather_repository, station_offset=2)
    severity_repository.save_assessment(
        make_assessment(low_event_id, level=FireSeverityLevel.LOW, score=10.0),
        weather_observation_ids=(low_weather_id,),
        satellite_hotspot_ids=(low_hotspot_id,),
        selected_frp_hotspot_id=low_hotspot_id,
    )

    demands = builder.build((critical_event_id, low_event_id), ASSESSED_AT)

    by_event = {demand.fire_event_id: demand for demand in demands}
    assert (by_event[critical_event_id].minimum_resources, by_event[critical_event_id].desired_resources) == (3, 4)
    assert (by_event[low_event_id].minimum_resources, by_event[low_event_id].desired_resources) == (1, 1)


def test_builder_uses_injected_policy(severity_repository):
    custom_policy = SeverityDemandPolicy(
        minimum_by_level={
            FireSeverityLevel.LOW: 5,
            FireSeverityLevel.MODERATE: 5,
            FireSeverityLevel.HIGH: 5,
            FireSeverityLevel.CRITICAL: 5,
        },
        desired_by_level={
            FireSeverityLevel.LOW: 5,
            FireSeverityLevel.MODERATE: 5,
            FireSeverityLevel.HIGH: 5,
            FireSeverityLevel.CRITICAL: 5,
        },
        fallback_minimum_resources=5,
        fallback_desired_resources=5,
    )
    custom_builder = GlobalIncidentDemandBuilder(fire_severity_assessment_repository=severity_repository, policy=custom_policy)

    (demand,) = custom_builder.build((1,), ASSESSED_AT)

    assert (demand.minimum_resources, demand.desired_resources) == (5, 5)
    assert demand.policy_methodology == custom_policy.methodology
