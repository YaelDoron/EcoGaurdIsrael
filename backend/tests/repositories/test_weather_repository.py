"""Unit tests for WeatherRepository, using a SQLite in-memory database.

Real PostgreSQL/Neon behavior (as opposed to SQLite's approximation) is
additionally covered by tests/integration/test_neon_database.py.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from src.database.models.weather_observation_db import WeatherObservationDB
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import WeatherRepositoryError, WeatherStationNotStoredError
from src.repositories.weather_repository import WeatherRepository


@pytest.fixture
def repository(sqlite_session_factory) -> WeatherRepository:
    return WeatherRepository(session_factory=sqlite_session_factory)


def make_station(**overrides) -> WeatherStation:
    defaults = dict(
        external_station_id=17, name="HAIFA", latitude=32.79, longitude=34.99, region_id=3, active=True
    )
    defaults.update(overrides)
    return WeatherStation(**defaults)


def make_observation(**overrides) -> WeatherObservation:
    defaults = dict(
        station_external_id=17,
        timestamp=datetime(2026, 9, 2, 12, 30, 0),
        temperature=31.4,
        relative_humidity=42,
        wind_speed=5.8,
        wind_direction=240,
        wind_gust=8.2,
        rainfall=0,
    )
    defaults.update(overrides)
    return WeatherObservation(**defaults)


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------


def test_save_station_inserts_new_station(repository):
    saved = repository.save_station(make_station())

    assert saved.external_station_id == 17
    assert len(repository.get_all_stations()) == 1


def test_save_station_same_external_id_does_not_create_duplicate(repository):
    repository.save_station(make_station())
    repository.save_station(make_station(name="HAIFA UPDATED"))

    assert len(repository.get_all_stations()) == 1


def test_save_station_updates_existing_metadata(repository):
    repository.save_station(make_station())

    updated = repository.save_station(
        make_station(name="HAIFA 2", latitude=33.0, longitude=35.0, region_id=9, active=False)
    )

    assert updated.name == "HAIFA 2"
    assert updated.latitude == 33.0
    assert updated.longitude == 35.0
    assert updated.region_id == 9
    assert updated.active is False


def test_get_station_by_external_id_returns_correct_station(repository):
    repository.save_station(make_station())

    found = repository.get_station_by_external_id(17)

    assert found is not None
    assert found.external_station_id == 17
    assert found.name == "HAIFA"


def test_get_station_by_external_id_returns_none_for_unknown_station(repository):
    assert repository.get_station_by_external_id(999) is None


def test_get_all_stations_is_deterministically_ordered(repository):
    repository.save_station(make_station(external_station_id=21, name="B"))
    repository.save_station(make_station(external_station_id=5, name="A"))
    repository.save_station(make_station(external_station_id=17, name="C"))

    stations = repository.get_all_stations()

    assert [s.external_station_id for s in stations] == [5, 17, 21]


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Timezone fix regression tests: `timestamp` is now DateTime(timezone=True)
# (applied for consistency with the satellite hotspot fix - see
# SatelliteHotspotRepository's own equivalent tests - so the new Weather
# Activity Feed signal never inherits the same class of tz-naive bug).
# ---------------------------------------------------------------------------


def test_observation_timestamp_column_is_declared_timezone_aware():
    assert WeatherObservationDB.__table__.columns["timestamp"].type.timezone is True


def test_aware_observation_timestamp_round_trips_preserving_the_exact_instant(repository):
    repository.save_station(make_station())
    aware_instant = datetime(2026, 9, 5, 10, 12, 0, tzinfo=timezone.utc)

    repository.save_observation(make_observation(timestamp=aware_instant))

    stored = repository.get_latest_observation(17)
    assert stored.timestamp.tzinfo is not None
    assert stored.timestamp == aware_instant


def test_naive_observation_timestamp_is_normalized_to_utc(repository):
    repository.save_station(make_station())
    naive_value = datetime(2026, 9, 5, 10, 12, 0)

    repository.save_observation(make_observation(timestamp=naive_value))

    stored = repository.get_latest_observation(17)
    assert stored.timestamp.tzinfo is not None
    assert stored.timestamp == naive_value.replace(tzinfo=timezone.utc)


def test_save_observation_persists_valid_observation(repository):
    repository.save_station(make_station())

    result = repository.save_observation(make_observation())

    assert result.is_duplicate is False
    assert result.observation.station_external_id == 17
    assert result.observation.temperature == 31.4


def test_save_observation_links_to_correct_station(repository):
    repository.save_station(make_station(external_station_id=17))
    repository.save_station(make_station(external_station_id=21, name="SECOND"))

    repository.save_observation(make_observation(station_external_id=17))
    repository.save_observation(
        make_observation(station_external_id=21, timestamp=datetime(2026, 9, 2, 13, 0, 0))
    )

    assert repository.get_latest_observation(17).station_external_id == 17
    assert repository.get_latest_observation(21).station_external_id == 21


def test_save_observation_with_none_optional_measurements(repository):
    repository.save_station(make_station())

    result = repository.save_observation(
        make_observation(
            temperature=None,
            relative_humidity=None,
            wind_speed=None,
            wind_direction=None,
            wind_gust=None,
            rainfall=None,
        )
    )

    assert result.observation.temperature is None
    assert result.observation.relative_humidity is None
    assert result.observation.wind_speed is None
    assert result.observation.wind_direction is None
    assert result.observation.wind_gust is None
    assert result.observation.rainfall is None


def test_save_observation_same_station_and_timestamp_is_not_duplicated(repository):
    repository.save_station(make_station())

    first = repository.save_observation(make_observation(temperature=20.0))
    second = repository.save_observation(make_observation(temperature=99.0))

    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert second.observation.temperature == 20.0  # original value, not overwritten

    stored = repository.get_observations_for_station(17)
    assert len(stored) == 1
    assert stored[0].temperature == 20.0


def test_save_observation_same_timestamp_different_stations_is_allowed(repository):
    repository.save_station(make_station(external_station_id=17))
    repository.save_station(make_station(external_station_id=21, name="SECOND"))
    timestamp = datetime(2026, 9, 2, 12, 30, 0)

    repository.save_observation(make_observation(station_external_id=17, timestamp=timestamp))
    repository.save_observation(make_observation(station_external_id=21, timestamp=timestamp))

    assert len(repository.get_observations_for_station(17)) == 1
    assert len(repository.get_observations_for_station(21)) == 1


def test_save_observation_same_station_different_timestamps_is_allowed(repository):
    repository.save_station(make_station())

    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 12, 30, 0)))
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 13, 30, 0)))

    assert len(repository.get_observations_for_station(17)) == 2


def test_save_observation_unknown_station_raises_clear_repository_error(repository):
    with pytest.raises(WeatherStationNotStoredError):
        repository.save_observation(make_observation(station_external_id=999))


def test_get_latest_observation_returns_newest(repository):
    repository.save_station(make_station())
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 12, 0, 0), temperature=10.0))
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 14, 0, 0), temperature=30.0))
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 13, 0, 0), temperature=20.0))

    latest = repository.get_latest_observation(17)

    assert latest.timestamp == datetime(2026, 9, 2, 14, 0, 0, tzinfo=timezone.utc)
    assert latest.temperature == 30.0


def test_get_latest_observation_returns_none_for_station_with_no_observations(repository):
    repository.save_station(make_station())

    assert repository.get_latest_observation(17) is None


def test_get_observations_for_station_respects_limit(repository):
    repository.save_station(make_station())
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 12, 0, 0)))
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 13, 0, 0)))
    repository.save_observation(make_observation(timestamp=datetime(2026, 9, 2, 14, 0, 0)))

    latest_two = repository.get_observations_for_station(17, limit=2)

    assert [o.timestamp for o in latest_two] == [
        datetime(2026, 9, 2, 14, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 2, 13, 0, 0, tzinfo=timezone.utc),
    ]


def test_get_recent_observations_for_area_candidates_returns_persisted_ids_and_bbox_candidates(
    repository,
):
    repository.save_station(make_station(external_station_id=17, latitude=32.0, longitude=35.0))
    repository.save_station(make_station(external_station_id=21, name="NEAR", latitude=32.01, longitude=35.0))
    repository.save_station(make_station(external_station_id=99, name="FAR", latitude=34.0, longitude=35.0))
    repository.save_observation(
        make_observation(station_external_id=17, timestamp=datetime(2026, 9, 2, 12, 0, 0))
    )
    repository.save_observation(
        make_observation(station_external_id=21, timestamp=datetime(2026, 9, 2, 12, 5, 0))
    )
    repository.save_observation(
        make_observation(station_external_id=99, timestamp=datetime(2026, 9, 2, 12, 10, 0))
    )

    records = repository.get_recent_observations_for_area_candidates(
        latitude=32.0,
        longitude=35.0,
        radius_km=5,
        start_time=datetime(2026, 9, 2, 11, 59, 0),
        end_time=datetime(2026, 9, 2, 12, 6, 0),
    )

    assert [record.station.external_station_id for record in records] == [17, 21]
    assert all(record.station_id > 0 for record in records)
    assert all(record.observation_id > 0 for record in records)
    assert [record.observation.timestamp for record in records] == [
        datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 2, 12, 5, 0, tzinfo=timezone.utc),
    ]


def test_get_recent_observations_for_area_candidates_orders_newest_per_station_first(repository):
    repository.save_station(make_station(external_station_id=17, latitude=32.0, longitude=35.0))
    repository.save_observation(
        make_observation(station_external_id=17, timestamp=datetime(2026, 9, 2, 12, 0, 0))
    )
    repository.save_observation(
        make_observation(station_external_id=17, timestamp=datetime(2026, 9, 2, 12, 5, 0))
    )

    records = repository.get_recent_observations_for_area_candidates(
        latitude=32.0,
        longitude=35.0,
        radius_km=5,
        start_time=datetime(2026, 9, 2, 11, 59, 0),
        end_time=datetime(2026, 9, 2, 12, 6, 0),
    )

    assert [record.observation.timestamp for record in records] == [
        datetime(2026, 9, 2, 12, 5, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc),
    ]


# ---------------------------------------------------------------------------
# Errors / transactions
# ---------------------------------------------------------------------------


def test_invalid_external_station_id_raises_repository_error(repository):
    with pytest.raises(WeatherRepositoryError):
        repository.get_station_by_external_id(-1)


def test_save_observation_handles_integrity_error_from_concurrent_duplicate(repository, monkeypatch):
    repository.save_station(make_station())
    repository.save_observation(make_observation(temperature=11.0))

    # Simulate a race: pretend the existing-row check found nothing (as if
    # another process inserted the row between the check and the insert), so
    # the repository must recover via the database's own unique constraint
    # (IntegrityError) rather than creating a duplicate row.
    original_find_observation = WeatherRepository._find_observation
    call_count = {"n": 0}

    def fake_find_observation(session, station_id, timestamp):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None
        return original_find_observation(session, station_id, timestamp)

    monkeypatch.setattr(WeatherRepository, "_find_observation", staticmethod(fake_find_observation))

    result = repository.save_observation(make_observation(temperature=99.0))

    assert result.is_duplicate is True
    assert result.observation.temperature == 11.0
    assert len(repository.get_observations_for_station(17)) == 1


def test_repository_session_is_usable_after_a_rolled_back_write(repository, sqlite_session_factory):
    repository.save_station(make_station())

    raw_session = sqlite_session_factory()
    raw_session.add(WeatherObservationDB(station_id=999999, timestamp=datetime(2026, 9, 2, 15, 0, 0)))
    with pytest.raises(IntegrityError):
        raw_session.commit()
    raw_session.rollback()
    raw_session.close()

    # The repository still works normally afterwards - no broken session state.
    result = repository.save_observation(make_observation())
    assert result.observation.station_external_id == 17
