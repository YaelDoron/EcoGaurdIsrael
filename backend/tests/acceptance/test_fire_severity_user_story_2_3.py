"""Acceptance coverage for User Story 2.3 active wildfire severity assessment."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest
from sqlalchemy import select

from src.agents.analysis import FireDetectionAgent, FireSeverityAssessmentAgent
from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator, _classify_score
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.external.copernicus import CopernicusServiceUnavailableError
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityInput,
    FireSeverityLevel,
    SatelliteHotspot,
    VegetationData,
    WeatherObservation,
    WeatherStation,
    WildfireReport,
)
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_detection import FireDetectionEvidenceService
from src.services.fire_severity import FireSeverityInputService
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventType,
    SimulationScenario,
)
from src.simulation.analysis import SimulationFireSeverityCoordinator

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
CARMEL_LATITUDE = 32.731
CARMEL_LONGITUDE = 35.046
GOLAN_LATITUDE = 33.186
GOLAN_LONGITUDE = 35.570


class UnavailableVegetationProvider:
    def get_land_cover_statistics(self, latitude, longitude, radius_km):
        raise CopernicusServiceUnavailableError("vegetation unavailable")


class StaticVegetationProvider:
    def get_land_cover_statistics(self, latitude, longitude, radius_km):
        return object()


class StaticVegetationMapper:
    def __init__(self, vegetation_data: VegetationData) -> None:
        self.vegetation_data = vegetation_data

    def map_statistics(self, statistics):
        return self.vegetation_data


class SpySeverityAgent:
    def __init__(self) -> None:
        self.calls = []

    def assess(self, fire_event_id, assessed_at):
        self.calls.append({"fire_event_id": fire_event_id, "assessed_at": assessed_at})
        return StoredFireSeverityAssessment(
            assessment_id=fire_event_id + 10_000,
            assessment=FireSeverityAssessment(
                fire_event_id=fire_event_id,
                assessed_at=assessed_at,
                status=FireSeverityAssessmentStatus.VALID,
                score=70.0,
                level=FireSeverityLevel.HIGH,
                methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
                methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            ),
            weather_observation_ids=(1,),
            satellite_hotspot_ids=(2,),
            selected_frp_hotspot_id=2,
        )


class FakeFireEventRepository:
    def __init__(self, events=()) -> None:
        self.events = tuple(events)
        self.calls = []

    def get_active_events_near(self, latitude, longitude, radius_km, as_of):
        self.calls.append({"latitude": latitude, "longitude": longitude, "radius_km": radius_km, "as_of": as_of})
        return self.events


class FakeDetectionAgent:
    def __init__(self, result: FireDetectionResult) -> None:
        self.result = result
        self.calls = []

    def detect(self, as_of):
        self.calls.append(as_of)
        return self.result


def severity_input(frp=50.0, wind=20.0, rh=40.0, vegetation=None) -> FireSeverityInput:
    return FireSeverityInput(
        frp_mw=frp,
        wind_speed_kmh=wind,
        relative_humidity_pct=rh,
        vegetation_fuel_score=vegetation,
    )


def make_severity_agent(
    sqlite_session_factory,
    *,
    vegetation_data: VegetationData | None = None,
    vegetation_unavailable: bool = True,
) -> FireSeverityAssessmentAgent:
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    if vegetation_data is None or vegetation_unavailable:
        provider = UnavailableVegetationProvider()
        mapper = StaticVegetationMapper(
            VegetationData(
                fuel_score=0.0,
                dominant_land_cover="unused",
                source="unused",
                dataset_year=2019,
                radius_km=1,
            )
        )
    else:
        provider = StaticVegetationProvider()
        mapper = StaticVegetationMapper(vegetation_data)

    return FireSeverityAssessmentAgent(
        input_service=FireSeverityInputService(
            fire_event_repository=fire_event_repository,
            weather_repository=weather_repository,
            satellite_hotspot_repository=satellite_repository,
            land_cover_client=provider,
            vegetation_mapper=mapper,
        ),
        calculator=FireSeverityCalculator(),
        repository=FireSeverityAssessmentRepository(session_factory=sqlite_session_factory),
    )


def make_detection_agent(sqlite_session_factory) -> FireDetectionAgent:
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news_repository = NewsRepository(session_factory=sqlite_session_factory)
    return FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(
            satellite_repository=satellite_repository,
            news_repository=news_repository,
        ),
        calculator=FireDetectionCalculator(),
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )


def save_weather(
    sqlite_session_factory,
    *,
    external_station_id: int,
    timestamp: datetime = AS_OF,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    wind_speed: float | None = 25.0,
    relative_humidity: float | None = 30.0,
) -> int:
    repository = WeatherRepository(session_factory=sqlite_session_factory)
    repository.save_station(
        WeatherStation(
            external_station_id=external_station_id,
            name=f"Severity Acceptance Station {external_station_id}",
            latitude=latitude,
            longitude=longitude,
        )
    )
    repository.save_observation(
        WeatherObservation(
            station_external_id=external_station_id,
            timestamp=timestamp,
            temperature=30.0,
            relative_humidity=relative_humidity,
            wind_speed=wind_speed,
        )
    )
    session = sqlite_session_factory()
    observation_id = (
        session.execute(
            select(WeatherObservationDB.id)
            .join(WeatherStationDB, WeatherObservationDB.station_id == WeatherStationDB.id)
            .where(WeatherStationDB.external_station_id == external_station_id)
            .order_by(WeatherObservationDB.id.desc())
        )
        .scalars()
        .first()
    )
    session.close()
    return observation_id


def save_satellite(
    sqlite_session_factory,
    *,
    detected_at: datetime = AS_OF,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    frp: float | None = 80.0,
) -> int:
    repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    satellite_name = f"ACCEPTANCE-{detected_at.timestamp()}-{frp}"
    repository.save_hotspot(
        SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            confidence="h",
            frp=frp,
            satellite=satellite_name,
            instrument="VIIRS",
        )
    )
    session = sqlite_session_factory()
    hotspot_id = (
        session.execute(
            select(SatelliteHotspotDB.id)
            .where(SatelliteHotspotDB.satellite == satellite_name)
            .order_by(SatelliteHotspotDB.id.desc())
        )
        .scalars()
        .first()
    )
    session.close()
    return hotspot_id


def save_news(
    sqlite_session_factory,
    *,
    source_url: str,
    published_at: datetime = AS_OF,
) -> int:
    repository = NewsRepository(session_factory=sqlite_session_factory)
    repository.save_report(
        WildfireReport(
            source_url=source_url,
            source_feed="Severity Acceptance",
            title="Wildfire report",
            summary="Visible smoke and flames.",
            location_name="Carmel",
            latitude=CARMEL_LATITUDE,
            longitude=CARMEL_LONGITUDE,
            published_at=published_at,
            fetched_at=published_at + timedelta(minutes=1),
        )
    )
    return repository.get_recent_reports(as_of=published_at + timedelta(minutes=2), lookback_minutes=10)[0].id


def create_fire_event(
    sqlite_session_factory,
    *,
    status: FireEventStatus = FireEventStatus.SUSPECTED,
    hotspot_id: int | None = None,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    detected_at: datetime = AS_OF - timedelta(minutes=5),
    updated_at: datetime = AS_OF - timedelta(minutes=5),
) -> StoredFireEvent:
    if hotspot_id is None:
        hotspot_id = save_satellite(sqlite_session_factory, detected_at=detected_at, latitude=latitude, longitude=longitude)
    return FireEventRepository(session_factory=sqlite_session_factory).create_event(
        FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            updated_at=updated_at,
            status=status,
            detection_confidence=0.7 if status is FireEventStatus.SUSPECTED else 0.85,
            methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
            methodology_version="1.0",
        ),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )


def test_stronger_fire_conditions_produce_higher_severity():
    calculator = FireSeverityCalculator()

    stronger = calculator.calculate(severity_input(frp=120, wind=45, rh=12))
    milder = calculator.calculate(severity_input(frp=20, wind=5, rh=80))

    assert stronger.score > milder.score
    assert _level_rank(stronger.level) >= _level_rank(milder.level)


def test_missing_required_weather_does_not_produce_valid_severity(sqlite_session_factory):
    hotspot_id = save_satellite(sqlite_session_factory, frp=90)
    event = create_fire_event(sqlite_session_factory, hotspot_id=hotspot_id)
    save_weather(sqlite_session_factory, external_station_id=970001, wind_speed=None, relative_humidity=20)

    result = make_severity_agent(sqlite_session_factory).assess(event.id, AS_OF)

    assert result.assessment.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert result.assessment.score is None
    assert result.assessment.level is None


def test_missing_required_frp_does_not_produce_valid_severity(sqlite_session_factory):
    hotspot_id = save_satellite(sqlite_session_factory, frp=None)
    event = create_fire_event(sqlite_session_factory, hotspot_id=hotspot_id)
    save_weather(sqlite_session_factory, external_station_id=970002, wind_speed=25, relative_humidity=20)

    result = make_severity_agent(sqlite_session_factory).assess(event.id, AS_OF)

    assert result.assessment.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert result.assessment.score is None
    assert result.assessment.level is None


def test_missing_optional_vegetation_still_allows_valid_assessment(sqlite_session_factory):
    hotspot_id = save_satellite(sqlite_session_factory, frp=70)
    event = create_fire_event(sqlite_session_factory, hotspot_id=hotspot_id)
    save_weather(sqlite_session_factory, external_station_id=970003, wind_speed=25, relative_humidity=30)

    result = make_severity_agent(sqlite_session_factory, vegetation_unavailable=True).assess(event.id, AS_OF)

    assert result.assessment.status is FireSeverityAssessmentStatus.VALID
    assert result.assessment.vegetation_source is None
    assert result.assessment.vegetation_dataset_year is None
    assert result.assessment.vegetation_radius_km is None
    assert result.assessment.vegetation_dominant_land_cover is None
    assert result.assessment.vegetation_fuel_score is None
    assert 0 <= result.assessment.score <= 100


def test_vegetation_is_optional_but_used_without_zero_penalty():
    calculator = FireSeverityCalculator()
    with_high_vegetation = calculator.calculate(severity_input(frp=40, wind=10, rh=60, vegetation=1.0))
    without_vegetation = calculator.calculate(severity_input(frp=40, wind=10, rh=60, vegetation=None))

    assert with_high_vegetation.level is _classify_score(with_high_vegetation.score)
    assert without_vegetation.level is _classify_score(without_vegetation.score)
    assert with_high_vegetation.vegetation_factor == 1.0
    assert without_vegetation.vegetation_factor is None
    assert without_vegetation.available_weight == pytest.approx(0.90)
    assert with_high_vegetation.score > without_vegetation.score


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (0.0, FireSeverityLevel.LOW),
        (24.999, FireSeverityLevel.LOW),
        (25.0, FireSeverityLevel.MODERATE),
        (49.999, FireSeverityLevel.MODERATE),
        (50.0, FireSeverityLevel.HIGH),
        (74.999, FireSeverityLevel.HIGH),
        (75.0, FireSeverityLevel.CRITICAL),
        (100.0, FireSeverityLevel.CRITICAL),
    ],
)
def test_level_boundaries_are_deterministic(score, expected_level):
    assert _classify_score(score) is expected_level


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_fire_event_does_not_receive_valid_severity(sqlite_session_factory, status):
    hotspot_id = save_satellite(sqlite_session_factory, frp=70)
    event = create_fire_event(sqlite_session_factory, hotspot_id=hotspot_id)
    FireEventRepository(session_factory=sqlite_session_factory).update_event(
        event.id,
        FireEvent(
            latitude=event.event.latitude,
            longitude=event.event.longitude,
            detected_at=event.event.detected_at,
            updated_at=AS_OF,
            status=status,
            detection_confidence=event.event.detection_confidence,
            methodology=event.event.methodology,
            methodology_version=event.event.methodology_version,
        ),
    )
    save_weather(sqlite_session_factory, external_station_id=970004, wind_speed=30, relative_humidity=25)

    result = make_severity_agent(sqlite_session_factory).assess(event.id, AS_OF)

    assert result.assessment.status is FireSeverityAssessmentStatus.INACTIVE_EVENT
    assert result.assessment.score is None
    assert result.assessment.level is None


def test_same_inputs_produce_same_severity():
    calculator = FireSeverityCalculator()
    first = calculator.calculate(severity_input(frp=55, wind=18, rh=35, vegetation=0.6))
    second = calculator.calculate(severity_input(frp=55, wind=18, rh=35, vegetation=0.6))

    assert second == first


def test_traceability_and_vegetation_snapshot_explain_persisted_assessment(sqlite_session_factory):
    first_weather_id = save_weather(sqlite_session_factory, external_station_id=970101, wind_speed=20, relative_humidity=35)
    second_weather_id = save_weather(sqlite_session_factory, external_station_id=970102, wind_speed=30, relative_humidity=25)
    lower_frp_id = save_satellite(sqlite_session_factory, detected_at=AS_OF - timedelta(minutes=1), frp=40)
    selected_frp_id = save_satellite(sqlite_session_factory, detected_at=AS_OF - timedelta(minutes=2), frp=95)
    event = create_fire_event(sqlite_session_factory, hotspot_id=lower_frp_id)
    FireEventRepository(session_factory=sqlite_session_factory).attach_evidence(
        event.id,
        (FireEvidenceRef(FireEvidenceType.SATELLITE, selected_frp_id),),
    )
    vegetation = VegetationData(
        fuel_score=0.75,
        dominant_land_cover="Shrubland",
        source="copernicus_land_cover",
        dataset_year=2019,
        radius_km=1.0,
    )

    result = make_severity_agent(
        sqlite_session_factory,
        vegetation_data=vegetation,
        vegetation_unavailable=False,
    ).assess(event.id, AS_OF)

    repository = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    stored = repository.get_by_id(result.assessment_id)
    assert stored.weather_observation_ids == tuple(sorted((first_weather_id, second_weather_id)))
    assert stored.satellite_hotspot_ids == tuple(sorted((lower_frp_id, selected_frp_id)))
    assert stored.selected_frp_hotspot_id == selected_frp_id
    assert stored.assessment.vegetation_source == "copernicus_land_cover"
    assert stored.assessment.vegetation_dataset_year == 2019
    assert stored.assessment.vegetation_radius_km == pytest.approx(1.0)
    assert stored.assessment.vegetation_dominant_land_cover == "Shrubland"
    assert stored.assessment.vegetation_fuel_score == pytest.approx(0.75)


def test_history_is_preserved_for_changing_fire_severity(sqlite_session_factory):
    hotspot_id = save_satellite(sqlite_session_factory, frp=60)
    event = create_fire_event(sqlite_session_factory, hotspot_id=hotspot_id)
    save_weather(sqlite_session_factory, external_station_id=970201, timestamp=AS_OF, wind_speed=10, relative_humidity=60)
    agent = make_severity_agent(sqlite_session_factory)
    first = agent.assess(event.id, AS_OF)
    save_weather(
        sqlite_session_factory,
        external_station_id=970202,
        timestamp=AS_OF + timedelta(minutes=10),
        wind_speed=45,
        relative_humidity=15,
    )

    second = agent.assess(event.id, AS_OF + timedelta(minutes=10))

    repository = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    latest = repository.get_latest_for_event(event.id)
    assert first.assessment_id != second.assessment_id
    assert repository.get_by_id(first.assessment_id).assessment.assessed_at == AS_OF
    assert repository.get_by_id(second.assessment_id).assessment.assessed_at == AS_OF + timedelta(minutes=10)
    assert latest.assessment_id == second.assessment_id


def test_weather_update_triggers_reassessment_for_nearby_active_fire_event():
    event_timestamp = AS_OF + timedelta(seconds=65)
    severity_agent = SpySeverityAgent()
    repository = FakeFireEventRepository(events=(StoredFireEvent(10, _fire_event_at(CARMEL_LATITUDE, CARMEL_LONGITUDE)),))
    scenario = _single_incident_scenario(SimulationEventType.WEATHER, 65)
    event = scenario.events[0]

    result = SimulationFireSeverityCoordinator(severity_agent, repository).handle_event(
        scenario=scenario,
        event=event,
        execution_result=_execution_result(event),
        event_timestamp=event_timestamp,
    )

    assert result.triggered is True
    assert severity_agent.calls == [{"fire_event_id": 10, "assessed_at": event_timestamp}]


def test_satellite_update_triggers_severity_after_detection_for_affected_event():
    event_timestamp = AS_OF + timedelta(seconds=20)
    detection_result = FireDetectionResult(True, 1, 0, 1, 0, (77,))
    detection_agent = FakeDetectionAgent(detection_result)
    severity_agent = SpySeverityAgent()
    scenario = _single_incident_scenario(SimulationEventType.SATELLITE, 20)
    event = scenario.events[0]
    execution_result = _execution_result(event)

    detected = detection_agent.detect(event_timestamp)
    severity = SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()).handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result,
        event_timestamp=event_timestamp,
        detection_result=detected,
    )

    assert detection_agent.calls == [event_timestamp]
    assert severity.triggered is True
    assert severity_agent.calls == [{"fire_event_id": 77, "assessed_at": event_timestamp}]


def test_news_detection_update_does_not_trigger_severity():
    event_timestamp = AS_OF + timedelta(seconds=40)
    detection_result = FireDetectionResult(True, 1, 0, 0, 1, (77,))
    detection_agent = FakeDetectionAgent(detection_result)
    severity_agent = SpySeverityAgent()
    scenario = _single_incident_scenario(SimulationEventType.NEWS, 40)
    event = scenario.events[0]

    detected = detection_agent.detect(event_timestamp)
    severity = SimulationFireSeverityCoordinator(severity_agent, FakeFireEventRepository()).handle_event(
        scenario=scenario,
        event=event,
        execution_result=_execution_result(event),
        event_timestamp=event_timestamp,
        detection_result=detected,
    )

    assert detection_agent.calls == [event_timestamp]
    assert severity.triggered is False
    assert severity_agent.calls == []


def test_weather_reassessment_does_not_cross_active_incident_areas():
    event_timestamp = AS_OF + timedelta(seconds=65)
    severity_agent = SpySeverityAgent()
    repository = FakeFireEventRepository(events=(StoredFireEvent(10, _fire_event_at(CARMEL_LATITUDE, CARMEL_LONGITUDE)),))
    scenario = _single_incident_scenario(SimulationEventType.WEATHER, 65)

    SimulationFireSeverityCoordinator(severity_agent, repository).handle_event(
        scenario=scenario,
        event=scenario.events[0],
        execution_result=_execution_result(scenario.events[0]),
        event_timestamp=event_timestamp,
    )

    assert repository.calls[0]["latitude"] == CARMEL_LOCATION.latitude
    assert severity_agent.calls == [{"fire_event_id": 10, "assessed_at": event_timestamp}]
    assert all(call["fire_event_id"] != 20 for call in severity_agent.calls)


def test_severity_does_not_use_fire_danger_score_or_news_confidence():
    calculator = FireSeverityCalculator()
    baseline = calculator.calculate(severity_input(frp=50, wind=20, rh=40))
    with_changed_unrelated_context = calculator.calculate(severity_input(frp=50, wind=20, rh=40))

    assert baseline == with_changed_unrelated_context


def test_persisted_valid_assessment_contains_methodology_metadata(sqlite_session_factory):
    hotspot_id = save_satellite(sqlite_session_factory, frp=70)
    event = create_fire_event(sqlite_session_factory, hotspot_id=hotspot_id)
    save_weather(sqlite_session_factory, external_station_id=970301, wind_speed=25, relative_humidity=30)

    result = make_severity_agent(sqlite_session_factory).assess(event.id, AS_OF)

    assert result.assessment.status is FireSeverityAssessmentStatus.VALID
    assert result.assessment.methodology == FIRE_SEVERITY_METHODOLOGY_NAME
    assert result.assessment.methodology_version == FIRE_SEVERITY_METHODOLOGY_VERSION


@pytest.mark.parametrize(
    "input_data",
    [
        severity_input(frp=0, wind=0, rh=100),
        severity_input(frp=100, wind=50, rh=0),
        severity_input(frp=1_000, wind=500, rh=0, vegetation=1),
        severity_input(frp=10, wind=5, rh=90, vegetation=0),
    ],
)
def test_valid_edge_inputs_keep_score_within_range(input_data):
    result = FireSeverityCalculator().calculate(input_data)

    assert 0 <= result.score <= 100
    assert math.isfinite(result.score)


def test_active_fire_simulation_trigger_sequence_is_weather_satellite_weather_satellite_only():
    severity_agent = SpySeverityAgent()
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(
            SimulatedIncident("incident-carmel-01", ScenarioType.ACTIVE_FIRE, CARMEL_LOCATION),
        ),
        events=(
            SimulationEvent(0, SimulationEventType.WEATHER, "incident-carmel-01", 0),
            SimulationEvent(20, SimulationEventType.SATELLITE, "incident-carmel-01", 0),
            SimulationEvent(40, SimulationEventType.NEWS, "incident-carmel-01", 0),
            SimulationEvent(65, SimulationEventType.WEATHER, "incident-carmel-01", 1),
            SimulationEvent(80, SimulationEventType.SATELLITE, "incident-carmel-01", 1),
            SimulationEvent(100, SimulationEventType.NEWS, "incident-carmel-01", 1),
        ),
    )
    repository = FakeFireEventRepository(events=())
    coordinator = SimulationFireSeverityCoordinator(severity_agent, repository)
    detection_results = {
        20: FireDetectionResult(True, 1, 0, 1, 0, (10,)),
        80: FireDetectionResult(True, 1, 0, 0, 1, (10,)),
    }

    for event in scenario.events:
        if event.offset_seconds == 65:
            repository.events = (StoredFireEvent(10, _fire_event_at(CARMEL_LATITUDE, CARMEL_LONGITUDE)),)
        coordinator.handle_event(
            scenario=scenario,
            event=event,
            execution_result=_execution_result(event),
            event_timestamp=AS_OF + timedelta(seconds=event.offset_seconds),
            detection_result=detection_results.get(event.offset_seconds),
        )

    assert severity_agent.calls == [
        {"fire_event_id": 10, "assessed_at": AS_OF + timedelta(seconds=20)},
        {"fire_event_id": 10, "assessed_at": AS_OF + timedelta(seconds=65)},
        {"fire_event_id": 10, "assessed_at": AS_OF + timedelta(seconds=80)},
    ]


def _fire_event_at(latitude: float, longitude: float) -> FireEvent:
    return FireEvent(
        latitude=latitude,
        longitude=longitude,
        detected_at=AS_OF,
        updated_at=AS_OF,
        status=FireEventStatus.SUSPECTED,
        detection_confidence=0.6,
        methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version="1.0",
    )


def _single_incident_scenario(event_type: SimulationEventType, offset_seconds: int) -> SimulationScenario:
    return SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(SimulatedIncident("incident-carmel-01", ScenarioType.ACTIVE_FIRE, CARMEL_LOCATION),),
        events=(SimulationEvent(offset_seconds, event_type, "incident-carmel-01", 0),),
    )


def _execution_result(event: SimulationEvent) -> SimulationEventExecutionResult:
    return SimulationEventExecutionResult(
        event=event,
        success=True,
        generated_count=1,
        saved_count=1,
        duplicates_skipped=0,
        failed_count=0,
    )


def _level_rank(level: FireSeverityLevel) -> int:
    return {
        FireSeverityLevel.LOW: 0,
        FireSeverityLevel.MODERATE: 1,
        FireSeverityLevel.HIGH: 2,
        FireSeverityLevel.CRITICAL: 3,
    }[level]
