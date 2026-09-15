"""Tests for WeatherRepository.get_observations_by_ids(), added for User Story
4.2 (wildfire-spread prediction) as an additive-only repository method.

Uses the same SQLite-in-memory fixture/helper conventions as
test_weather_repository.py, kept in a separate file per Task 5 ownership
rules rather than editing the existing teammate-owned test file.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import WeatherRepositoryError
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


def _persist_observation_and_get_id(
    repository: WeatherRepository,
    *,
    external_station_id: int = 17,
    latitude: float = 32.79,
    longitude: float = 34.99,
    timestamp: datetime = datetime(2026, 9, 2, 12, 30, 0),
) -> int:
    """Save a station+observation and recover its DB id via an existing read method."""
    repository.save_station(make_station(external_station_id=external_station_id, latitude=latitude, longitude=longitude))
    repository.save_observation(
        make_observation(station_external_id=external_station_id, timestamp=timestamp)
    )

    candidates = repository.get_recent_observations_for_area_candidates(
        latitude=latitude,
        longitude=longitude,
        radius_km=1.0,
        start_time=timestamp,
        end_time=timestamp,
    )
    return candidates[0].observation_id


def test_single_id_returns_observation_with_station_info(repository):
    observation_id = _persist_observation_and_get_id(repository)

    results = repository.get_observations_by_ids((observation_id,))

    assert len(results) == 1
    assert results[0].observation_id == observation_id
    assert results[0].station.external_station_id == 17
    assert results[0].observation.temperature == 31.4
    assert results[0].observation.wind_speed == 5.8


def test_multiple_ids_returns_all_matching_observations(repository):
    first_id = _persist_observation_and_get_id(
        repository, external_station_id=17, latitude=32.79, longitude=34.99, timestamp=datetime(2026, 9, 2, 12, 0, 0)
    )
    second_id = _persist_observation_and_get_id(
        repository, external_station_id=21, latitude=32.80, longitude=35.0, timestamp=datetime(2026, 9, 2, 13, 0, 0)
    )

    results = repository.get_observations_by_ids((first_id, second_id))

    assert {record.observation_id for record in results} == {first_id, second_id}


def test_results_are_deterministically_ordered_by_ascending_id(repository):
    first_id = _persist_observation_and_get_id(
        repository, external_station_id=17, latitude=32.79, longitude=34.99, timestamp=datetime(2026, 9, 2, 12, 0, 0)
    )
    second_id = _persist_observation_and_get_id(
        repository, external_station_id=21, latitude=32.80, longitude=35.0, timestamp=datetime(2026, 9, 2, 13, 0, 0)
    )

    results_forward = repository.get_observations_by_ids((first_id, second_id))
    results_reversed_request = repository.get_observations_by_ids((second_id, first_id))

    expected_order = sorted([first_id, second_id])
    assert [record.observation_id for record in results_forward] == expected_order
    assert [record.observation_id for record in results_reversed_request] == expected_order


def test_duplicate_requested_ids_return_single_result(repository):
    observation_id = _persist_observation_and_get_id(repository)

    results = repository.get_observations_by_ids((observation_id, observation_id, observation_id))

    assert len(results) == 1
    assert results[0].observation_id == observation_id


def test_missing_id_is_silently_omitted(repository):
    observation_id = _persist_observation_and_get_id(repository)
    missing_id = observation_id + 999

    results = repository.get_observations_by_ids((observation_id, missing_id))

    returned_ids = {record.observation_id for record in results}
    assert returned_ids == {observation_id}
    # Caller can detect the missing id by diffing requested vs. returned.
    assert missing_id not in returned_ids


def test_all_ids_missing_returns_empty_tuple(repository):
    results = repository.get_observations_by_ids((123456,))
    assert results == ()


def test_empty_input_returns_empty_tuple_without_error(repository):
    assert repository.get_observations_by_ids(()) == ()


@pytest.mark.parametrize("invalid_id", [0, -1, True, "1"])
def test_invalid_id_raises(repository, invalid_id):
    with pytest.raises(WeatherRepositoryError):
        repository.get_observations_by_ids((invalid_id,))


def test_reconstructs_full_observation_and_station_fields(repository):
    repository.save_station(make_station(external_station_id=55, name="TIBERIAS", latitude=32.79, longitude=35.53))
    repository.save_observation(
        make_observation(
            station_external_id=55,
            timestamp=datetime(2026, 9, 2, 14, 0, 0),
            temperature=33.0,
            relative_humidity=25,
            wind_speed=12.5,
            wind_direction=90,
            wind_gust=15.0,
            rainfall=0.0,
        )
    )
    candidates = repository.get_recent_observations_for_area_candidates(
        latitude=32.79, longitude=35.53, radius_km=1.0,
        start_time=datetime(2026, 9, 2, 14, 0, 0), end_time=datetime(2026, 9, 2, 14, 0, 0),
    )
    observation_id = candidates[0].observation_id

    result = repository.get_observations_by_ids((observation_id,))[0]

    assert result.station.name == "TIBERIAS"
    assert result.station.latitude == 32.79
    assert result.station.longitude == 35.53
    assert result.observation.station_external_id == 55
    assert result.observation.temperature == 33.0
    assert result.observation.relative_humidity == 25
    assert result.observation.wind_speed == 12.5
    assert result.observation.wind_direction == 90
    assert result.observation.wind_gust == 15.0
    assert result.observation.rainfall == 0.0
