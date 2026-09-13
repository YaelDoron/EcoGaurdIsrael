"""Unit tests for FireDangerAssessmentRepository using SQLite in-memory DB."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.calculators.fire_danger.ffwi_config import (
    FFWI_METHODOLOGY_NAME,
    FFWI_METHODOLOGY_VERSION,
)
from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.models import FireDangerAssessment, FireDangerAssessmentStatus, FireDangerLevel
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import FireDangerAssessmentRepositoryError
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.weather_repository import WeatherRepository

ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def repository(sqlite_session_factory) -> FireDangerAssessmentRepository:
    return FireDangerAssessmentRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def weather_repository(sqlite_session_factory) -> WeatherRepository:
    return WeatherRepository(session_factory=sqlite_session_factory)


def make_assessment(**overrides) -> FireDangerAssessment:
    defaults = dict(
        area_id="area-carmel",
        area_name="Carmel",
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


def create_weather_observation(weather_repository: WeatherRepository, external_station_id: int) -> tuple[int, int]:
    weather_repository.save_station(
        WeatherStation(
            external_station_id=external_station_id,
            name=f"Station {external_station_id}",
            latitude=32.7,
            longitude=35.0,
        )
    )
    timestamp = datetime(2026, 9, 14, 11, external_station_id % 60, tzinfo=timezone.utc)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=external_station_id,
            timestamp=timestamp,
            temperature=30,
            relative_humidity=25,
            wind_speed=12,
        )
    )
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.7,
        longitude=35.0,
        radius_km=10,
        start_time=timestamp - timedelta(minutes=1),
        end_time=timestamp + timedelta(minutes=1),
    )
    record = next(
        candidate
        for candidate in candidates
        if candidate.station.external_station_id == external_station_id
    )
    return record.observation_id, record.station_id


def get_trace_rows(sqlite_session_factory, assessment_id: int):
    session = sqlite_session_factory()
    rows = (
        session.execute(
            select(FireDangerAssessmentWeatherInputDB)
            .where(FireDangerAssessmentWeatherInputDB.assessment_id == assessment_id)
            .order_by(FireDangerAssessmentWeatherInputDB.id.asc())
        )
        .scalars()
        .all()
    )
    session.close()
    return rows


def count_assessments(sqlite_session_factory) -> int:
    session = sqlite_session_factory()
    count = len(session.execute(select(FireDangerAssessmentDB)).scalars().all())
    session.close()
    return count


def test_valid_assessment_saves_successfully(repository, weather_repository):
    observation_id, station_id = create_weather_observation(weather_repository, 1001)

    saved = repository.save_assessment(make_assessment(), (observation_id,), (station_id,))

    assert saved.assessment_id > 0
    assert saved.assessment == make_assessment()
    assert saved.observation_ids == (observation_id,)
    assert saved.station_ids == (station_id,)


def test_persisted_assessment_fields_match(repository, weather_repository, sqlite_session_factory):
    observation_id, station_id = create_weather_observation(weather_repository, 1001)

    saved = repository.save_assessment(make_assessment(), (observation_id,), (station_id,))

    session = sqlite_session_factory()
    db_assessment = session.get(FireDangerAssessmentDB, saved.assessment_id)
    session.close()
    assert db_assessment.area_id == "area-carmel"
    assert db_assessment.area_name == "Carmel"
    assert db_assessment.area_latitude == 32.731
    assert db_assessment.area_longitude == 35.046
    assert db_assessment.area_radius_km == 5.0
    assert db_assessment.assessed_at.replace(tzinfo=timezone.utc) == ASSESSED_AT
    assert db_assessment.score == 42.5
    assert db_assessment.danger_level == FireDangerLevel.VERY_HIGH.value
    assert db_assessment.status == FireDangerAssessmentStatus.VALID.value
    assert db_assessment.methodology == FFWI_METHODOLOGY_NAME
    assert db_assessment.methodology_version == FFWI_METHODOLOGY_VERSION


def test_valid_assessment_creates_trace_rows(repository, weather_repository, sqlite_session_factory):
    first_observation_id, first_station_id = create_weather_observation(weather_repository, 1001)
    second_observation_id, second_station_id = create_weather_observation(weather_repository, 1002)

    saved = repository.save_assessment(
        make_assessment(),
        (first_observation_id, second_observation_id),
        (first_station_id, second_station_id),
    )

    trace_rows = get_trace_rows(sqlite_session_factory, saved.assessment_id)
    assert len(trace_rows) == 2


def test_trace_rows_reference_correct_weather_observation_ids(
    repository,
    weather_repository,
    sqlite_session_factory,
):
    first_observation_id, first_station_id = create_weather_observation(weather_repository, 1001)
    second_observation_id, second_station_id = create_weather_observation(weather_repository, 1002)

    saved = repository.save_assessment(
        make_assessment(),
        (first_observation_id, second_observation_id),
        (first_station_id, second_station_id),
    )

    trace_rows = get_trace_rows(sqlite_session_factory, saved.assessment_id)
    assert [row.weather_observation_id for row in trace_rows] == [
        first_observation_id,
        second_observation_id,
    ]


def test_trace_rows_preserve_station_ids(repository, weather_repository, sqlite_session_factory):
    first_observation_id, first_station_id = create_weather_observation(weather_repository, 1001)
    second_observation_id, second_station_id = create_weather_observation(weather_repository, 1002)

    saved = repository.save_assessment(
        make_assessment(),
        (first_observation_id, second_observation_id),
        (first_station_id, second_station_id),
    )

    trace_rows = get_trace_rows(sqlite_session_factory, saved.assessment_id)
    assert [row.station_id for row in trace_rows] == [first_station_id, second_station_id]


def test_mismatched_traceability_lengths_are_rejected(repository, weather_repository):
    observation_id, station_id = create_weather_observation(weather_repository, 1001)

    with pytest.raises(FireDangerAssessmentRepositoryError):
        repository.save_assessment(make_assessment(), (observation_id,), (station_id, station_id))


def test_valid_assessment_with_empty_trace_source_collections_is_rejected(repository):
    with pytest.raises(FireDangerAssessmentRepositoryError):
        repository.save_assessment(make_assessment(), (), ())


def test_insufficient_data_saves_without_score_level_or_trace_rows(
    repository,
    sqlite_session_factory,
):
    assessment = make_assessment(
        status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
    )

    saved = repository.save_assessment(assessment, (), ())

    assert saved.observation_ids == ()
    assert saved.station_ids == ()
    session = sqlite_session_factory()
    db_assessment = session.get(FireDangerAssessmentDB, saved.assessment_id)
    trace_count = len(
        session.execute(
            select(FireDangerAssessmentWeatherInputDB).where(
                FireDangerAssessmentWeatherInputDB.assessment_id == saved.assessment_id
            )
        )
        .scalars()
        .all()
    )
    session.close()
    assert db_assessment.score is None
    assert db_assessment.danger_level is None
    assert trace_count == 0


def test_duplicate_weather_observation_within_same_assessment_is_rejected_and_rolled_back(
    repository,
    weather_repository,
    sqlite_session_factory,
):
    observation_id, station_id = create_weather_observation(weather_repository, 1001)

    with pytest.raises(FireDangerAssessmentRepositoryError):
        repository.save_assessment(
            make_assessment(),
            (observation_id, observation_id),
            (station_id, station_id),
        )

    assert count_assessments(sqlite_session_factory) == 0


def test_get_by_id_returns_expected_stored_assessment(repository, weather_repository):
    observation_id, station_id = create_weather_observation(weather_repository, 1001)
    saved = repository.save_assessment(make_assessment(), (observation_id,), (station_id,))

    found = repository.get_by_id(saved.assessment_id)

    assert found == saved


def test_get_latest_for_area_orders_by_assessed_at(repository, weather_repository):
    observation_id, station_id = create_weather_observation(weather_repository, 1001)
    older = make_assessment(assessed_at=ASSESSED_AT - timedelta(hours=1), score=20, level=FireDangerLevel.MODERATE)
    newer = make_assessment(assessed_at=ASSESSED_AT, score=30, level=FireDangerLevel.HIGH)
    repository.save_assessment(older, (observation_id,), (station_id,))
    saved_newer = repository.save_assessment(newer, (observation_id,), (station_id,))

    latest = repository.get_latest_for_area("area-carmel")

    assert latest == saved_newer


def test_transaction_rolls_back_if_trace_persistence_fails(repository, sqlite_session_factory):
    with pytest.raises(FireDangerAssessmentRepositoryError):
        repository.save_assessment(make_assessment(), (999999,), (1,))

    assert count_assessments(sqlite_session_factory) == 0
    session = sqlite_session_factory()
    assert session.execute(select(WeatherObservationDB)).scalars().all() == []
    session.close()
