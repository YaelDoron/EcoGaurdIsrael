"""Live integration test for fire-danger assessment persistence against Neon."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.calculators.fire_danger.ffwi_config import (
    FFWI_METHODOLOGY_NAME,
    FFWI_METHODOLOGY_VERSION,
)
from src.config.settings import settings
from src.database.connection import get_engine, get_session, init_db
from src.models import FireDangerAssessment, FireDangerAssessmentStatus, FireDangerLevel
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.weather_repository import WeatherRepository

pytestmark = pytest.mark.integration

VALID_AREA_ID = "ecoguard-integration-fire-danger-valid"
INSUFFICIENT_AREA_ID = "ecoguard-integration-fire-danger-insufficient"
TEST_STATION_IDS = (988001, 988002)
ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


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
                "DELETE FROM fire_danger_assessment_weather_inputs "
                "WHERE assessment_id IN ("
                "SELECT id FROM fire_danger_assessments "
                "WHERE area_id IN (:valid_area_id, :insufficient_area_id)"
                ")"
            ),
            {"valid_area_id": VALID_AREA_ID, "insufficient_area_id": INSUFFICIENT_AREA_ID},
        )
        connection.execute(
            text(
                "DELETE FROM fire_danger_assessments "
                "WHERE area_id IN (:valid_area_id, :insufficient_area_id)"
            ),
            {"valid_area_id": VALID_AREA_ID, "insufficient_area_id": INSUFFICIENT_AREA_ID},
        )
        connection.execute(
            text(
                "DELETE FROM weather_observations "
                "WHERE station_id IN ("
                "SELECT id FROM weather_stations "
                "WHERE external_station_id IN (:first_station_id, :second_station_id)"
                ")"
            ),
            {"first_station_id": TEST_STATION_IDS[0], "second_station_id": TEST_STATION_IDS[1]},
        )
        connection.execute(
            text(
                "DELETE FROM weather_stations "
                "WHERE external_station_id IN (:first_station_id, :second_station_id)"
            ),
            {"first_station_id": TEST_STATION_IDS[0], "second_station_id": TEST_STATION_IDS[1]},
        )


def make_assessment(**overrides) -> FireDangerAssessment:
    defaults = dict(
        area_id=VALID_AREA_ID,
        area_name="Integration Fire Danger Area",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=ASSESSED_AT,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireDangerAssessment(**defaults)


def _create_weather_inputs() -> tuple[tuple[int, ...], tuple[int, ...]]:
    repository = WeatherRepository()
    for index, external_station_id in enumerate(TEST_STATION_IDS):
        repository.save_station(
            WeatherStation(
                external_station_id=external_station_id,
                name=f"ECOGUARD_FIRE_DANGER_INTEGRATION_{external_station_id}",
                latitude=32.731 + index * 0.001,
                longitude=35.046,
            )
        )
        repository.save_observation(
            WeatherObservation(
                station_external_id=external_station_id,
                timestamp=ASSESSED_AT - timedelta(minutes=5 - index),
                temperature=30 + index,
                relative_humidity=25 + index,
                wind_speed=12 + index,
            )
        )

    candidates = repository.get_recent_observations_for_area_candidates(
        latitude=32.731,
        longitude=35.046,
        radius_km=5,
        start_time=ASSESSED_AT - timedelta(minutes=10),
        end_time=ASSESSED_AT,
    )
    selected = [
        candidate
        for candidate in candidates
        if candidate.station.external_station_id in TEST_STATION_IDS
    ]
    selected.sort(key=lambda candidate: candidate.station.external_station_id)
    return (
        tuple(candidate.observation_id for candidate in selected),
        tuple(candidate.station_id for candidate in selected),
    )


def test_valid_fire_danger_assessment_saves_reads_and_preserves_traceability_against_neon():
    init_db()
    observation_ids, station_ids = _create_weather_inputs()
    repository = FireDangerAssessmentRepository()

    saved = repository.save_assessment(make_assessment(), observation_ids, station_ids)
    found = repository.get_by_id(saved.assessment_id)

    assert found == saved
    assert found.assessment.methodology == FFWI_METHODOLOGY_NAME
    assert found.assessment.methodology_version == FFWI_METHODOLOGY_VERSION
    assert found.observation_ids == observation_ids
    assert found.station_ids == station_ids

    with get_session() as session:
        trace_rows = (
            session.execute(
                text(
                    "SELECT weather_observation_id, station_id "
                    "FROM fire_danger_assessment_weather_inputs "
                    "WHERE assessment_id = :assessment_id "
                    "ORDER BY id"
                ),
                {"assessment_id": saved.assessment_id},
            )
            .mappings()
            .all()
        )

    assert [row["weather_observation_id"] for row in trace_rows] == list(observation_ids)
    assert [row["station_id"] for row in trace_rows] == list(station_ids)


def test_insufficient_data_fire_danger_assessment_saves_null_result_and_no_trace_rows_against_neon():
    init_db()
    repository = FireDangerAssessmentRepository()
    assessment = make_assessment(
        area_id=INSUFFICIENT_AREA_ID,
        status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
    )

    saved = repository.save_assessment(assessment, (), ())
    found = repository.get_by_id(saved.assessment_id)

    assert found.assessment.score is None
    assert found.assessment.level is None
    assert found.observation_ids == ()
    assert found.station_ids == ()

    with get_session() as session:
        db_values = session.execute(
            text(
                "SELECT score, danger_level FROM fire_danger_assessments "
                "WHERE id = :assessment_id"
            ),
            {"assessment_id": saved.assessment_id},
        ).one()
        trace_count = session.execute(
            text(
                "SELECT COUNT(*) FROM fire_danger_assessment_weather_inputs "
                "WHERE assessment_id = :assessment_id"
            ),
            {"assessment_id": saved.assessment_id},
        ).scalar_one()

    assert db_values.score is None
    assert db_values.danger_level is None
    assert trace_count == 0
