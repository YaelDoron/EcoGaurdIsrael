"""Live integration test against the real Neon PostgreSQL database.

Run explicitly (not part of the default unit-test run):

    python -m pytest -m integration -v

Automatically skipped if DATABASE_URL is not configured, so the regular
unit-test run (`pytest -m "not integration"`) never depends on Neon being
reachable.

This test only ever touches its own clearly-identifiable temporary rows
(external_station_id=999999 / name="ECOGUARD_INTEGRATION_TEST_STATION") and
deletes them at the start and end of every test. It never drops tables,
drops the database, modifies unrelated rows, or prints DATABASE_URL / any
credentials.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from src.config.settings import settings
from src.database.connection import get_engine, get_session, init_db
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.weather_repository import WeatherRepository

pytestmark = pytest.mark.integration

TEST_EXTERNAL_STATION_ID = 999999
TEST_STATION_NAME = "ECOGUARD_INTEGRATION_TEST_STATION"
# UTC-aware: weather_observations.timestamp is TIMESTAMPTZ (scripts/migrate_satellite_and_weather_timestamps.py)
TEST_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _delete_test_rows() -> None:
    """Delete only this test's own rows, identified by TEST_EXTERNAL_STATION_ID."""
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM weather_observations "
                "WHERE station_id IN ("
                "SELECT id FROM weather_stations WHERE external_station_id = :external_station_id"
                ")"
            ),
            {"external_station_id": TEST_EXTERNAL_STATION_ID},
        )
        connection.execute(
            text("DELETE FROM weather_stations WHERE external_station_id = :external_station_id"),
            {"external_station_id": TEST_EXTERNAL_STATION_ID},
        )


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url) -> None:
    """Guarantee no leftover EcoGuard integration-test rows before or after each test."""
    init_db()  # ensure tables exist before attempting to clean rows from them
    _delete_test_rows()
    yield
    _delete_test_rows()


def test_can_connect_and_select_1() -> None:
    with get_session() as session:
        result = session.execute(text("SELECT 1")).scalar_one()

    assert result == 1


def test_init_db_creates_expected_tables() -> None:
    init_db()

    with get_session() as session:
        table_names = (
            session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' "
                    "AND table_name IN ('weather_stations', 'weather_observations')"
                )
            )
            .scalars()
            .all()
        )

    assert set(table_names) == {"weather_stations", "weather_observations"}


def test_insert_read_and_duplicate_prevention_against_real_neon() -> None:
    init_db()
    repository = WeatherRepository()

    saved_station = repository.save_station(
        WeatherStation(
            external_station_id=TEST_EXTERNAL_STATION_ID,
            name=TEST_STATION_NAME,
            latitude=32.0,
            longitude=34.8,
            region_id=1,
            active=True,
        )
    )
    assert saved_station.external_station_id == TEST_EXTERNAL_STATION_ID
    assert saved_station.name == TEST_STATION_NAME

    fetched_station = repository.get_station_by_external_id(TEST_EXTERNAL_STATION_ID)
    assert fetched_station is not None
    assert fetched_station.name == TEST_STATION_NAME
    assert fetched_station.latitude == 32.0
    assert fetched_station.longitude == 34.8

    save_result = repository.save_observation(
        WeatherObservation(
            station_external_id=TEST_EXTERNAL_STATION_ID,
            timestamp=TEST_TIMESTAMP,
            temperature=25.5,
            relative_humidity=50,
            wind_speed=3.0,
            wind_direction=180,
            wind_gust=6.0,
            rainfall=0,
        )
    )
    assert save_result.is_duplicate is False
    assert save_result.observation.station_external_id == TEST_EXTERNAL_STATION_ID
    assert save_result.observation.temperature == 25.5

    latest = repository.get_latest_observation(TEST_EXTERNAL_STATION_ID)
    assert latest is not None
    assert latest.timestamp == TEST_TIMESTAMP
    assert latest.temperature == 25.5

    # Duplicate prevention: re-saving the exact same (station, timestamp)
    # must not create a second row, and must return the originally stored
    # values rather than the new (deliberately different) ones.
    duplicate_attempt = repository.save_observation(
        WeatherObservation(
            station_external_id=TEST_EXTERNAL_STATION_ID,
            timestamp=TEST_TIMESTAMP,
            temperature=999.0,
        )
    )
    assert duplicate_attempt.is_duplicate is True
    assert duplicate_attempt.observation.temperature == 25.5

    all_observations = repository.get_observations_for_station(TEST_EXTERNAL_STATION_ID)
    assert len(all_observations) == 1


def test_cleanup_removes_all_temporary_integration_test_rows() -> None:
    init_db()
    repository = WeatherRepository()
    repository.save_station(
        WeatherStation(
            external_station_id=TEST_EXTERNAL_STATION_ID,
            name=TEST_STATION_NAME,
            latitude=32.0,
            longitude=34.8,
        )
    )
    repository.save_observation(
        WeatherObservation(station_external_id=TEST_EXTERNAL_STATION_ID, timestamp=TEST_TIMESTAMP)
    )

    _delete_test_rows()

    assert repository.get_station_by_external_id(TEST_EXTERNAL_STATION_ID) is None
    assert repository.get_latest_observation(TEST_EXTERNAL_STATION_ID) is None
