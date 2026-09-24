"""Acceptance coverage for User Story 2.2 active wildfire detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.agents.analysis import FireDangerAssessmentAgent, FireDetectionAgent
from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.models import (
    AssessmentArea,
    FireDangerAssessmentStatus,
    FireEventStatus,
    FireEvidenceType,
    SatelliteHotspot,
    WeatherObservation,
    WeatherStation,
    WildfireReport,
)
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger import FireDangerInputService
from src.services.fire_detection import FireDetectionEvidenceService
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    SimulationEventType,
    build_active_fire_scenario,
    simulation_event_timestamp,
)
from src.simulation.analysis import SimulationFireDetectionCoordinator

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
CARMEL_LATITUDE = 32.731
CARMEL_LONGITUDE = 35.046
GOLAN_LATITUDE = 33.186
GOLAN_LONGITUDE = 35.570


def make_detection_agent(sqlite_session_factory) -> FireDetectionAgent:
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news_repository = NewsRepository(session_factory=sqlite_session_factory)
    evidence_service = FireDetectionEvidenceService(
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    return FireDetectionAgent(
        evidence_service=evidence_service,
        calculator=FireDetectionCalculator(),
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
        satellite_repository=satellite_repository,
        news_repository=news_repository,
        # Pinned (Task 9C): these rule-behaviour tests must not depend on FIRE_DETECTION_DECISION_MODE in the environment.
        decision_policy=FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.SHADOW),
    )


def make_fire_danger_agent(sqlite_session_factory) -> FireDangerAssessmentAgent:
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    return FireDangerAssessmentAgent(
        input_service=FireDangerInputService(weather_repository=weather_repository),
        calculator=FFWICalculator(),
        repository=FireDangerAssessmentRepository(session_factory=sqlite_session_factory),
    )


def save_satellite(
    sqlite_session_factory,
    *,
    detected_at: datetime = AS_OF,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    confidence: str = "n",
) -> int:
    repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    repository.save_hotspot(
        SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            confidence=confidence,
            satellite="SIM-NOAA-20",
            instrument="VIIRS",
        )
    )
    return repository.get_recent_hotspots(as_of=detected_at + timedelta(minutes=1), lookback_minutes=10)[0].id


def save_news(
    sqlite_session_factory,
    *,
    source_url: str,
    published_at: datetime = AS_OF,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
) -> int:
    repository = NewsRepository(session_factory=sqlite_session_factory)
    repository.save_report(
        WildfireReport(
            source_url=source_url,
            source_feed="Acceptance",
            title="Wildfire reported",
            summary="Smoke and flames reported.",
            location_name="Acceptance Location",
            latitude=latitude,
            longitude=longitude,
            published_at=published_at,
            fetched_at=published_at + timedelta(minutes=1),
        )
    )
    return repository.get_recent_reports(as_of=published_at + timedelta(minutes=2), lookback_minutes=10)[0].id


def fire_events(sqlite_session_factory):
    session = sqlite_session_factory()
    rows = session.execute(select(FireEventDB).order_by(FireEventDB.id.asc())).scalars().all()
    session.close()
    return rows


def get_stored_event(sqlite_session_factory, event_id: int):
    return FireEventRepository(session_factory=sqlite_session_factory).get_by_id(event_id)


def test_matching_satellite_and_news_confirm_fire_event(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    save_news(sqlite_session_factory, source_url="https://acceptance.example/at1", published_at=AS_OF + timedelta(minutes=20))

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    assert result.events_created == 1
    stored = get_stored_event(sqlite_session_factory, result.event_ids[0])
    assert stored.event.status is FireEventStatus.CONFIRMED
    assert stored.event.detection_confidence == pytest.approx(0.80)
    assert {ref.evidence_type for ref in stored.supporting_evidence} == {
        FireEvidenceType.SATELLITE,
        FireEvidenceType.NEWS,
    }


def test_suspected_to_confirmed_upgrade_uses_same_fire_event(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    first = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=5))
    first_id = first.event_ids[0]
    first_event = get_stored_event(sqlite_session_factory, first_id)
    assert first_event.event.status is FireEventStatus.SUSPECTED
    assert first_event.event.detection_confidence == pytest.approx(0.60)

    save_news(sqlite_session_factory, source_url="https://acceptance.example/at1b", published_at=AS_OF + timedelta(minutes=20))
    second = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))

    assert second.events_created == 0
    assert second.events_updated == 1
    assert second.event_ids == (first_id,)
    upgraded = get_stored_event(sqlite_session_factory, first_id)
    assert upgraded.event.status is FireEventStatus.CONFIRMED
    assert upgraded.event.detection_confidence == pytest.approx(0.80)
    assert len(upgraded.supporting_evidence) == 2


def test_geographically_distant_evidence_is_not_combined(sqlite_session_factory):
    save_satellite(sqlite_session_factory, detected_at=AS_OF, latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)
    save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/at2-golan",
        published_at=AS_OF + timedelta(minutes=20),
        latitude=GOLAN_LATITUDE,
        longitude=GOLAN_LONGITUDE,
    )

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    assert result.events_created == 2
    events = fire_events(sqlite_session_factory)
    assert len(events) == 2
    assert all(event.status == FireEventStatus.SUSPECTED.value for event in events)
    assert all(event.detection_confidence < 0.80 for event in events)


def test_temporally_distant_evidence_is_not_combined(sqlite_session_factory):
    save_satellite(sqlite_session_factory, detected_at=AS_OF - timedelta(minutes=70), confidence="l")
    save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/at3",
        published_at=AS_OF,
    )

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=1))

    assert result.success is True
    assert result.no_event_count == 1
    assert result.events_created == 1
    stored = get_stored_event(sqlite_session_factory, result.event_ids[0])
    assert stored.event.status is FireEventStatus.SUSPECTED
    assert stored.event.detection_confidence == pytest.approx(0.50)
    assert {ref.evidence_type for ref in stored.supporting_evidence} == {FireEvidenceType.NEWS}


def test_temporal_correlation_boundary_is_inclusive_only_at_sixty_minutes(sqlite_session_factory):
    save_satellite(sqlite_session_factory, detected_at=AS_OF, confidence="n")
    save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/at3-boundary",
        published_at=AS_OF + timedelta(minutes=60),
    )

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=61))

    assert result.events_created == 1
    stored = get_stored_event(sqlite_session_factory, result.event_ids[0])
    assert stored.event.status is FireEventStatus.CONFIRMED
    assert stored.event.detection_confidence == pytest.approx(0.80)


def test_duplicate_satellite_reprocessing_does_not_create_second_event(sqlite_session_factory):
    hotspot_id = save_satellite(sqlite_session_factory, detected_at=AS_OF, confidence="n")
    first = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=5))

    second = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=5))

    assert second.events_created == 0
    assert second.events_updated == 0
    assert second.event_ids == first.event_ids
    assert len(fire_events(sqlite_session_factory)) == 1
    session = sqlite_session_factory()
    satellite_links = session.execute(select(FireEventSatelliteEvidenceDB)).scalars().all()
    session.close()
    assert [(link.fire_event_id, link.satellite_hotspot_id) for link in satellite_links] == [
        (first.event_ids[0], hotspot_id)
    ]
    assert get_stored_event(sqlite_session_factory, first.event_ids[0]).event.detection_confidence == pytest.approx(0.60)


def test_single_nominal_satellite_creates_suspected_event(sqlite_session_factory):
    save_satellite(sqlite_session_factory, detected_at=AS_OF, confidence="n")

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=5))

    stored = get_stored_event(sqlite_session_factory, result.event_ids[0])
    assert stored.event.status is FireEventStatus.SUSPECTED
    assert stored.event.detection_confidence == pytest.approx(0.60)


def test_high_fire_danger_without_direct_evidence_does_not_create_fire_event(sqlite_session_factory):
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    weather_repository.save_station(
        WeatherStation(
            external_station_id=990001,
            name="Acceptance High Fire Danger Station",
            latitude=CARMEL_LATITUDE,
            longitude=CARMEL_LONGITUDE,
        )
    )
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=990001,
            timestamp=AS_OF,
            temperature=43.0,
            relative_humidity=6.0,
            wind_speed=55.0,
        )
    )
    danger_result = make_fire_danger_agent(sqlite_session_factory).assess(
        area=AssessmentArea(
            id="acceptance-fire-detection-separation",
            name="Acceptance Fire Danger Area",
            latitude=CARMEL_LATITUDE,
            longitude=CARMEL_LONGITUDE,
            radius_km=5.0,
        ),
        as_of=AS_OF,
    )
    detection_result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF)

    assert danger_result.success is True
    assert danger_result.assessment.status is FireDangerAssessmentStatus.VALID
    assert detection_result.success is True
    assert detection_result.candidates_processed == 0
    assert fire_events(sqlite_session_factory) == []


def test_reprocessing_same_confirmed_evidence_is_idempotent(sqlite_session_factory):
    save_satellite(sqlite_session_factory, detected_at=AS_OF, confidence="n")
    save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/at7",
        published_at=AS_OF + timedelta(minutes=20),
    )
    first = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))
    event_id = first.event_ids[0]
    before = get_stored_event(sqlite_session_factory, event_id)

    second = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))
    after = get_stored_event(sqlite_session_factory, event_id)

    assert second.events_created == 0
    assert second.events_updated == 0
    assert second.event_ids == (event_id,)
    assert len(fire_events(sqlite_session_factory)) == 1
    assert after.event == before.event
    assert after.supporting_evidence == before.supporting_evidence


def test_source_aware_evidence_refs_preserve_overlapping_numeric_ids(sqlite_session_factory):
    satellite_id = save_satellite(sqlite_session_factory, detected_at=AS_OF, confidence="n")
    news_id = save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/source-aware",
        published_at=AS_OF + timedelta(minutes=20),
    )
    assert satellite_id == news_id

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))

    refs = get_stored_event(sqlite_session_factory, result.event_ids[0]).supporting_evidence
    assert {(ref.evidence_type, ref.evidence_id) for ref in refs} == {
        (FireEvidenceType.SATELLITE, satellite_id),
        (FireEvidenceType.NEWS, news_id),
    }


def test_two_geographically_separate_wildfires_create_two_logical_fire_events(sqlite_session_factory):
    save_satellite(sqlite_session_factory, detected_at=AS_OF, confidence="n", latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)
    save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/carmel",
        published_at=AS_OF + timedelta(minutes=20),
        latitude=CARMEL_LATITUDE,
        longitude=CARMEL_LONGITUDE,
    )
    save_satellite(sqlite_session_factory, detected_at=AS_OF + timedelta(minutes=1), confidence="n", latitude=GOLAN_LATITUDE, longitude=GOLAN_LONGITUDE)
    save_news(
        sqlite_session_factory,
        source_url="https://acceptance.example/golan",
        published_at=AS_OF + timedelta(minutes=21),
        latitude=GOLAN_LATITUDE,
        longitude=GOLAN_LONGITUDE,
    )

    result = make_detection_agent(sqlite_session_factory).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    assert result.events_created == 2
    stored_events = [get_stored_event(sqlite_session_factory, event_id) for event_id in result.event_ids]
    assert all(stored.event.status is FireEventStatus.CONFIRMED for stored in stored_events)
    assert all(stored.event.detection_confidence == pytest.approx(0.80) for stored in stored_events)
    latitudes = sorted(stored.event.latitude for stored in stored_events)
    assert latitudes == pytest.approx(sorted([CARMEL_LATITUDE, GOLAN_LATITUDE]))


def test_simulation_active_fire_satellite_then_news_updates_same_event(sqlite_session_factory):
    scenario = build_active_fire_scenario(seed=42)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news_repository = NewsRepository(session_factory=sqlite_session_factory)
    executor = SimulationEventExecutor(
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    coordinator = SimulationFireDetectionCoordinator(detection_agent=make_detection_agent(sqlite_session_factory))
    started_at = AS_OF
    satellite_event = next(event for event in scenario.events if event.event_type is SimulationEventType.SATELLITE)
    news_event = next(event for event in scenario.events if event.event_type is SimulationEventType.NEWS)

    satellite_timestamp = simulation_event_timestamp(started_at, satellite_event)
    satellite_execution: SimulationEventExecutionResult = executor.execute(
        scenario=scenario,
        event=satellite_event,
        event_timestamp=satellite_timestamp,
    )
    satellite_detection = coordinator.handle_event(
        scenario=scenario,
        event=satellite_event,
        execution_result=satellite_execution,
        event_timestamp=satellite_timestamp,
    )
    assert satellite_detection.triggered is True
    assert satellite_detection.detection_result.success is True
    assert satellite_detection.detection_result.events_created == 1
    event_id = satellite_detection.detection_result.event_ids[0]

    news_timestamp = simulation_event_timestamp(started_at, news_event)
    news_execution: SimulationEventExecutionResult = executor.execute(
        scenario=scenario,
        event=news_event,
        event_timestamp=news_timestamp,
    )
    news_detection = coordinator.handle_event(
        scenario=scenario,
        event=news_event,
        execution_result=news_execution,
        event_timestamp=news_timestamp,
    )

    assert news_detection.triggered is True
    assert news_detection.detection_result.events_created == 0
    assert news_detection.detection_result.event_ids == (event_id,)
    stored = get_stored_event(sqlite_session_factory, event_id)
    assert stored.event.status is FireEventStatus.CONFIRMED
    assert len(stored.supporting_evidence) == 2
