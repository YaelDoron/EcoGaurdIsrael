"""Tests for the SQLAlchemy ORM models: table structure and constraints.

Uses SQLite in-memory (see tests/conftest.py) to verify NOT NULL, UNIQUE and
FOREIGN KEY constraints at the schema level. SQLite approximates PostgreSQL
but is not identical; real PostgreSQL behavior is additionally covered by
tests/integration/test_neon_database.py.
"""
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB

TIMESTAMP = datetime(2026, 9, 2, 12, 30, 0)


def _insert_station(session_factory, **overrides) -> int:
    defaults = dict(
        external_station_id=17, name="HAIFA", latitude=32.79, longitude=34.99, region_id=3, active=True
    )
    defaults.update(overrides)
    session = session_factory()
    station = WeatherStationDB(**defaults)
    session.add(station)
    session.commit()
    station_id = station.id
    session.close()
    return station_id


def test_table_names():
    assert WeatherStationDB.__tablename__ == "weather_stations"
    assert WeatherObservationDB.__tablename__ == "weather_observations"
    assert FireDangerAssessmentDB.__tablename__ == "fire_danger_assessments"
    assert FireDangerAssessmentWeatherInputDB.__tablename__ == (
        "fire_danger_assessment_weather_inputs"
    )
    assert ResponseTargetSetDB.__tablename__ == "response_target_sets"
    assert ResponseTargetDB.__tablename__ == "response_targets"


def test_response_target_tables_registered_in_metadata():
    table_names = set(ResponseTargetSetDB.metadata.tables)

    assert "response_target_sets" in table_names
    assert "response_targets" in table_names


def test_response_target_order_unique_constraint_present():
    constraint_names = {constraint.name for constraint in ResponseTargetDB.__table__.constraints}

    assert "uq_response_targets_set_order" in constraint_names
    assert "ck_response_targets_type_source_metadata" in constraint_names


def test_station_primary_key_autoincrements(sqlite_session_factory):
    session = sqlite_session_factory()
    station = WeatherStationDB(external_station_id=17, name="HAIFA", latitude=32.79, longitude=34.99)
    session.add(station)
    session.commit()

    assert isinstance(station.id, int)
    session.close()


def test_observation_primary_key_autoincrements(sqlite_session_factory):
    station_id = _insert_station(sqlite_session_factory)
    session = sqlite_session_factory()
    observation = WeatherObservationDB(station_id=station_id, timestamp=TIMESTAMP)
    session.add(observation)
    session.commit()

    assert isinstance(observation.id, int)
    session.close()


def test_external_station_id_is_unique(sqlite_session_factory):
    _insert_station(sqlite_session_factory, external_station_id=17)

    session = sqlite_session_factory()
    session.add(WeatherStationDB(external_station_id=17, name="OTHER", latitude=1, longitude=1))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_station_and_timestamp_unique_constraint(sqlite_session_factory):
    station_id = _insert_station(sqlite_session_factory)

    session = sqlite_session_factory()
    session.add(WeatherObservationDB(station_id=station_id, timestamp=TIMESTAMP))
    session.commit()
    session.close()

    session2 = sqlite_session_factory()
    session2.add(WeatherObservationDB(station_id=station_id, timestamp=TIMESTAMP))
    with pytest.raises(IntegrityError):
        session2.commit()
    session2.rollback()
    session2.close()


def test_foreign_key_is_enforced_for_unknown_station(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(WeatherObservationDB(station_id=999999, timestamp=TIMESTAMP))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_optional_weather_fields_can_be_none(sqlite_session_factory):
    station_id = _insert_station(sqlite_session_factory)
    session = sqlite_session_factory()
    observation = WeatherObservationDB(station_id=station_id, timestamp=TIMESTAMP)
    session.add(observation)
    session.commit()

    assert observation.temperature is None
    assert observation.relative_humidity is None
    assert observation.wind_speed is None
    assert observation.wind_direction is None
    assert observation.wind_gust is None
    assert observation.rainfall is None
    session.close()


def test_station_name_is_required(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(WeatherStationDB(external_station_id=17, name=None, latitude=32.79, longitude=34.99))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_station_latitude_is_required(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(WeatherStationDB(external_station_id=17, name="HAIFA", latitude=None, longitude=34.99))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_observation_timestamp_is_required(sqlite_session_factory):
    station_id = _insert_station(sqlite_session_factory)
    session = sqlite_session_factory()
    session.add(WeatherObservationDB(station_id=station_id, timestamp=None))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_observation_station_id_is_required(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(WeatherObservationDB(station_id=None, timestamp=TIMESTAMP))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_station_relationship_exposes_its_observations(sqlite_session_factory):
    station_id = _insert_station(sqlite_session_factory)
    session = sqlite_session_factory()
    session.add(WeatherObservationDB(station_id=station_id, timestamp=TIMESTAMP))
    session.commit()

    station = session.get(WeatherStationDB, station_id)
    assert len(station.observations) == 1
    assert station.observations[0].timestamp == TIMESTAMP
    session.close()


def test_fire_danger_assessment_trace_observation_unique_per_assessment(sqlite_session_factory):
    station_id = _insert_station(sqlite_session_factory)
    session = sqlite_session_factory()
    observation = WeatherObservationDB(station_id=station_id, timestamp=TIMESTAMP)
    assessment = FireDangerAssessmentDB(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=TIMESTAMP,
        status="valid",
        score=42.5,
        danger_level="very_high",
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    session.add_all([observation, assessment])
    session.commit()
    session.add(
        FireDangerAssessmentWeatherInputDB(
            assessment_id=assessment.id,
            weather_observation_id=observation.id,
            station_id=station_id,
        )
    )
    session.commit()

    session.add(
        FireDangerAssessmentWeatherInputDB(
            assessment_id=assessment.id,
            weather_observation_id=observation.id,
            station_id=station_id,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_fire_danger_assessment_trace_requires_existing_observation(sqlite_session_factory):
    session = sqlite_session_factory()
    assessment = FireDangerAssessmentDB(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=TIMESTAMP,
        status="valid",
        score=42.5,
        danger_level="very_high",
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    session.add(assessment)
    session.commit()
    session.add(
        FireDangerAssessmentWeatherInputDB(
            assessment_id=assessment.id,
            weather_observation_id=999999,
            station_id=1,
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()
