"""Live integration test for FireSeverityAssessment persistence against Neon."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.config.settings import settings
from src.database.connection import get_engine, init_db
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
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository

pytestmark = pytest.mark.integration

ASSESSED_AT = datetime(2026, 3, 23, 3, 23, 23, tzinfo=timezone.utc)
DETECTED_AT = ASSESSED_AT - timedelta(minutes=30)
LATITUDE = 31.34567
LONGITUDE = 35.87654
STATION_EXTERNAL_ID = 923923


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url) -> None:
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM fire_severity_assessment_weather_inputs "
                "WHERE assessment_id IN ("
                "SELECT id FROM fire_severity_assessments "
                "WHERE methodology = :methodology AND assessed_at IN (:assessed_at, :second_assessed_at)"
                ")"
            ),
            {
                "methodology": FIRE_SEVERITY_METHODOLOGY_NAME,
                "assessed_at": ASSESSED_AT,
                "second_assessed_at": ASSESSED_AT + timedelta(minutes=15),
            },
        )
        connection.execute(
            text(
                "DELETE FROM fire_severity_assessment_satellite_inputs "
                "WHERE assessment_id IN ("
                "SELECT id FROM fire_severity_assessments "
                "WHERE methodology = :methodology AND assessed_at IN (:assessed_at, :second_assessed_at)"
                ")"
            ),
            {
                "methodology": FIRE_SEVERITY_METHODOLOGY_NAME,
                "assessed_at": ASSESSED_AT,
                "second_assessed_at": ASSESSED_AT + timedelta(minutes=15),
            },
        )
        connection.execute(
            text(
                "DELETE FROM fire_severity_assessments "
                "WHERE methodology = :methodology AND assessed_at IN (:assessed_at, :second_assessed_at)"
            ),
            {
                "methodology": FIRE_SEVERITY_METHODOLOGY_NAME,
                "assessed_at": ASSESSED_AT,
                "second_assessed_at": ASSESSED_AT + timedelta(minutes=15),
            },
        )
        connection.execute(
            text(
                "DELETE FROM fire_event_satellite_evidence "
                "WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology = :methodology AND detected_at = :detected_at"
                ")"
            ),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": DETECTED_AT},
        )
        connection.execute(
            text("DELETE FROM fire_events WHERE methodology = :methodology AND detected_at = :detected_at"),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": DETECTED_AT},
        )
        connection.execute(
            text(
                "DELETE FROM weather_observations "
                "WHERE station_id IN (SELECT id FROM weather_stations WHERE external_station_id = :station_id)"
            ),
            {"station_id": STATION_EXTERNAL_ID},
        )
        connection.execute(
            text("DELETE FROM weather_stations WHERE external_station_id = :station_id"),
            {"station_id": STATION_EXTERNAL_ID},
        )
        connection.execute(
            text(
                "DELETE FROM satellite_hotspots "
                "WHERE detected_at IN (:detected_at, :second_detected_at) "
                "AND latitude = :latitude AND longitude = :longitude"
            ),
            {
                "detected_at": DETECTED_AT,
                "second_detected_at": DETECTED_AT + timedelta(minutes=1),
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
            },
        )


def make_event() -> FireEvent:
    return FireEvent(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        detected_at=DETECTED_AT,
        updated_at=ASSESSED_AT - timedelta(minutes=5),
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )


def make_assessment(fire_event_id: int, **overrides) -> FireSeverityAssessment:
    defaults = dict(
        fire_event_id=fire_event_id,
        assessed_at=ASSESSED_AT,
        status=FireSeverityAssessmentStatus.VALID,
        score=76.5,
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


def _create_source_rows() -> tuple[int, tuple[int, int], tuple[int, int]]:
    satellite_repository = SatelliteHotspotRepository()
    weather_repository = WeatherRepository()

    weather_repository.save_station(
        WeatherStation(
            external_station_id=STATION_EXTERNAL_ID,
            name="Severity Integration Station",
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )
    )
    first_observation = WeatherObservation(
        station_external_id=STATION_EXTERNAL_ID,
        timestamp=ASSESSED_AT - timedelta(minutes=10),
        temperature=31.0,
        relative_humidity=25.0,
        wind_speed=18.0,
    )
    second_observation = WeatherObservation(
        station_external_id=STATION_EXTERNAL_ID,
        timestamp=ASSESSED_AT - timedelta(minutes=5),
        temperature=32.0,
        relative_humidity=20.0,
        wind_speed=24.0,
    )
    weather_repository.save_observation(first_observation)
    weather_repository.save_observation(second_observation)
    weather_ids = tuple(
        sorted(
            record.observation_id
            for record in weather_repository.get_recent_observations_for_area_candidates(
                latitude=LATITUDE,
                longitude=LONGITUDE,
                radius_km=5.0,
                start_time=ASSESSED_AT - timedelta(minutes=30),
                end_time=ASSESSED_AT,
            )
        )
    )

    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=DETECTED_AT,
            confidence="h",
            frp=40.0,
            satellite="N20",
        )
    )
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=DETECTED_AT + timedelta(minutes=1),
            confidence="h",
            frp=80.0,
            satellite="N20",
        )
    )
    satellite_ids = tuple(
        sorted(
            record.id
            for record in satellite_repository.get_recent_hotspots(
                as_of=ASSESSED_AT,
                lookback_minutes=120,
            )
        )
    )
    selected_hotspot_id = max(
        (satellite_repository.get_by_id(hotspot_id) for hotspot_id in satellite_ids),
        key=lambda record: record.hotspot.frp,
    ).id
    return selected_hotspot_id, weather_ids, satellite_ids


def test_fire_severity_assessment_repository_round_trip_history_and_traceability_against_neon():
    selected_hotspot_id, weather_ids, satellite_ids = _create_source_rows()
    fire_event_repository = FireEventRepository()
    fire_event = fire_event_repository.create_event(
        make_event(),
        (FireEvidenceRef(FireEvidenceType.SATELLITE, selected_hotspot_id),),
    )
    repository = FireSeverityAssessmentRepository()

    first = repository.save_assessment(
        make_assessment(fire_event.id),
        weather_observation_ids=weather_ids,
        satellite_hotspot_ids=satellite_ids,
        selected_frp_hotspot_id=selected_hotspot_id,
    )
    second = repository.save_assessment(
        make_assessment(
            fire_event.id,
            assessed_at=ASSESSED_AT + timedelta(minutes=15),
            score=88.0,
            level=FireSeverityLevel.CRITICAL,
        ),
        weather_observation_ids=weather_ids,
        satellite_hotspot_ids=satellite_ids,
        selected_frp_hotspot_id=selected_hotspot_id,
    )

    found = repository.get_by_id(first.assessment_id)
    latest = repository.get_latest_for_event(fire_event.id)

    assert found.assessment == first.assessment
    assert repository.get_weather_input_ids(first.assessment_id) == weather_ids
    assert repository.get_satellite_input_ids(first.assessment_id) == satellite_ids
    assert repository.get_selected_frp_hotspot_id(first.assessment_id) == selected_hotspot_id
    assert found.assessment.vegetation_source == "COPERNICUS_GLOBAL_LAND_COVER_100M_API"
    assert found.assessment.vegetation_dataset_year == 2019
    assert found.assessment.vegetation_radius_km == pytest.approx(1.0)
    assert found.assessment.vegetation_dominant_land_cover == "Tree cover"
    assert found.assessment.vegetation_fuel_score == pytest.approx(0.9)
    assert repository.get_by_id(first.assessment_id) is not None
    assert repository.get_by_id(second.assessment_id) is not None
    assert latest.assessment_id == second.assessment_id
    assert latest.assessment.score == pytest.approx(88.0)
