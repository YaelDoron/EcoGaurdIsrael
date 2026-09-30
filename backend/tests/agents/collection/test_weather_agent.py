"""Unit tests for WeatherAgent.

All dependencies (IMSClient, WeatherMapper, WeatherRepository) are mocked -
these tests require no IMS API token, no internet access, and no Neon
connection. Real PostgreSQL/Neon behavior is covered separately by
tests/integration/test_neon_database.py; real IMS behavior will be covered
once the real API token is available.
"""
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.agents.collection.weather_agent import WeatherAgent
from src.external.ims.exceptions import IMSServiceUnavailableError
from src.external.ims.ims_client import IMSClient
from src.mappers.exceptions import MissingRequiredWeatherFieldError, WeatherMappingError
from src.mappers.weather_mapper import WeatherMapper
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import WeatherRepositoryError, WeatherStationNotStoredError
from src.repositories.weather_repository import SaveObservationResult, WeatherRepository

TIMESTAMP = datetime(2026, 9, 2, 12, 30, 0)


def make_raw_station(station_id: int) -> dict:
    return {
        "stationId": station_id,
        "name": f"STATION_{station_id}",
        "location": {"latitude": 32.0, "longitude": 34.0},
    }


def make_raw_observation(station_id: int) -> dict:
    return {"stationId": station_id, "data": [{"datetime": "2026-09-02T12:30:00+03:00", "channels": []}]}


def make_domain_station(station_id: int) -> WeatherStation:
    return WeatherStation(
        external_station_id=station_id, name=f"STATION_{station_id}", latitude=32.0, longitude=34.0
    )


def make_domain_observation(station_id: int, **overrides) -> WeatherObservation:
    defaults = dict(station_external_id=station_id, timestamp=TIMESTAMP)
    defaults.update(overrides)
    return WeatherObservation(**defaults)


@pytest.fixture
def ims_client() -> Mock:
    return Mock(spec=IMSClient)


@pytest.fixture
def weather_mapper() -> Mock:
    return Mock(spec=WeatherMapper)


@pytest.fixture
def weather_repository() -> Mock:
    return Mock(spec=WeatherRepository)


@pytest.fixture
def agent(ims_client: Mock, weather_mapper: Mock, weather_repository: Mock) -> WeatherAgent:
    return WeatherAgent(
        ims_client=ims_client, weather_mapper=weather_mapper, weather_repository=weather_repository
    )


def _wire_happy_path(ims_client: Mock, weather_mapper: Mock, weather_repository: Mock) -> None:
    """Wire mocks so mapping/persistence always succeeds for any station id."""
    weather_mapper.map_station.side_effect = lambda raw: make_domain_station(raw["stationId"])
    weather_repository.save_station.side_effect = lambda station: station
    ims_client.get_station_data.side_effect = lambda station_id: make_raw_observation(station_id)
    weather_mapper.map_observation.side_effect = lambda raw: make_domain_observation(raw["stationId"])
    weather_repository.save_observation.side_effect = lambda obs: SaveObservationResult(
        observation=obs, is_duplicate=False
    )


# ---------------------------------------------------------------------------
# Happy path
# Acceptance test: "Valid observations returned by IMS are stored correctly."
# ---------------------------------------------------------------------------


def test_collect_happy_path_processes_and_stores_all_stations(agent, ims_client, weather_mapper, weather_repository):
    raw_stations = [make_raw_station(1), make_raw_station(2)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    result = agent.collect()

    ims_client.get_stations.assert_called_once()
    assert weather_mapper.map_station.call_count == 2
    assert weather_repository.save_station.call_count == 2
    assert ims_client.get_station_data.call_count == 2
    ims_client.get_station_data.assert_any_call(1)
    ims_client.get_station_data.assert_any_call(2)
    assert weather_mapper.map_observation.call_count == 2
    assert weather_repository.save_observation.call_count == 2

    assert result.success is True
    assert result.stations_received == 2
    assert result.stations_processed == 2
    assert result.stations_failed == 0
    assert result.observations_saved == 2
    assert result.duplicates_skipped == 0
    assert result.observations_failed == 0


def test_collect_uses_external_station_id_not_internal_db_id(agent, ims_client, weather_mapper, weather_repository):
    ims_client.get_stations.return_value = [make_raw_station(42)]
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    agent.collect()

    ims_client.get_station_data.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# Acceptance test: "If one station is unavailable, observations from other
# stations are still processed."
# ---------------------------------------------------------------------------


def test_collect_one_station_unavailable_does_not_stop_the_others(agent, ims_client, weather_mapper, weather_repository):
    raw_stations = [make_raw_station(1), make_raw_station(2), make_raw_station(3), make_raw_station(4)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    def get_station_data_side_effect(station_id):
        if station_id == 3:
            raise IMSServiceUnavailableError("IMS request timed out.")
        return make_raw_observation(station_id)

    ims_client.get_station_data.side_effect = get_station_data_side_effect

    result = agent.collect()

    # All 4 stations were attempted - station 3's failure did not skip station 4.
    assert ims_client.get_station_data.call_count == 4
    ims_client.get_station_data.assert_any_call(4)

    assert result.success is True
    assert result.stations_received == 4
    assert result.stations_processed == 4  # station metadata itself always saved fine
    assert result.stations_failed == 0
    assert result.observations_saved == 3  # stations 1, 2, 4
    assert result.observations_failed == 1  # station 3


# ---------------------------------------------------------------------------
# Acceptance test: "If a specific measurement is missing, the remaining
# measurements are stored."
# ---------------------------------------------------------------------------


def test_collect_missing_optional_measurement_still_saves_the_observation(
    agent, ims_client, weather_mapper, weather_repository
):
    ims_client.get_stations.return_value = [make_raw_station(1)]
    weather_mapper.map_station.return_value = make_domain_station(1)
    weather_repository.save_station.return_value = make_domain_station(1)
    ims_client.get_station_data.return_value = make_raw_observation(1)

    observation_missing_gust_and_rain = make_domain_observation(1, wind_gust=None, rainfall=None)
    weather_mapper.map_observation.return_value = observation_missing_gust_and_rain
    weather_repository.save_observation.return_value = SaveObservationResult(
        observation=observation_missing_gust_and_rain, is_duplicate=False
    )

    result = agent.collect()

    weather_repository.save_observation.assert_called_once_with(observation_missing_gust_and_rain)
    assert result.stations_failed == 0
    assert result.observations_failed == 0
    assert result.observations_saved == 1


# ---------------------------------------------------------------------------
# Acceptance test: "Repeated observations are not stored as duplicates."
# ---------------------------------------------------------------------------


def test_collect_duplicate_observation_is_a_normal_condition_not_a_failure(
    agent, ims_client, weather_mapper, weather_repository
):
    ims_client.get_stations.return_value = [make_raw_station(1)]
    domain_station = make_domain_station(1)
    weather_mapper.map_station.return_value = domain_station
    weather_repository.save_station.return_value = domain_station
    ims_client.get_station_data.return_value = make_raw_observation(1)
    domain_observation = make_domain_observation(1)
    weather_mapper.map_observation.return_value = domain_observation
    weather_repository.save_observation.return_value = SaveObservationResult(
        observation=domain_observation, is_duplicate=True
    )

    result = agent.collect()

    assert result.success is True
    assert result.stations_failed == 0
    assert result.observations_saved == 0
    assert result.duplicates_skipped == 1
    assert result.observations_failed == 0


# ---------------------------------------------------------------------------
# Acceptance test: "If IMS is temporarily unavailable, existing information
# is preserved and the system does not crash."
# ---------------------------------------------------------------------------


def test_collect_ims_entirely_unavailable_stops_before_any_repository_write(
    agent, ims_client, weather_mapper, weather_repository
):
    ims_client.get_stations.side_effect = IMSServiceUnavailableError("IMS service returned HTTP 503.")

    result = agent.collect()

    weather_mapper.map_station.assert_not_called()
    weather_repository.save_station.assert_not_called()
    ims_client.get_station_data.assert_not_called()
    weather_mapper.map_observation.assert_not_called()
    weather_repository.save_observation.assert_not_called()

    assert result.success is False
    assert result.stations_received == 0
    assert result.stations_processed == 0
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# Acceptance test: "An observation without a valid station identifier is
# handled without corrupting stored data."
# ---------------------------------------------------------------------------


def test_collect_invalid_station_is_skipped_without_requesting_its_weather_data(
    agent, ims_client, weather_mapper, weather_repository
):
    raw_stations = [make_raw_station(1), make_raw_station(2)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    def map_station_side_effect(raw):
        if raw["stationId"] == 1:
            raise MissingRequiredWeatherFieldError("Raw IMS station is missing required field 'name'.")
        return make_domain_station(raw["stationId"])

    weather_mapper.map_station.side_effect = map_station_side_effect

    result = agent.collect()

    # Station 1 (invalid) must never reach save_station or get_station_data.
    weather_repository.save_station.assert_called_once()
    assert weather_repository.save_station.call_args.args[0].external_station_id == 2
    ims_client.get_station_data.assert_called_once_with(2)

    assert result.success is True
    assert result.stations_failed == 1
    assert result.stations_processed == 1


# ---------------------------------------------------------------------------
# Station persistence failure
# ---------------------------------------------------------------------------


def test_collect_station_repository_failure_skips_its_observation_and_continues(
    agent, ims_client, weather_mapper, weather_repository
):
    raw_stations = [make_raw_station(1), make_raw_station(2)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    def save_station_side_effect(station):
        if station.external_station_id == 1:
            raise WeatherRepositoryError("DB write failed")
        return station

    weather_repository.save_station.side_effect = save_station_side_effect

    result = agent.collect()

    # Station 1's persistence failure must not trigger a data request for it.
    ims_client.get_station_data.assert_called_once_with(2)

    assert result.stations_failed == 1
    assert result.stations_processed == 1
    assert result.observations_saved == 1


# ---------------------------------------------------------------------------
# Observation mapping failure
# ---------------------------------------------------------------------------


def test_collect_observation_mapping_failure_keeps_the_station_stored(
    agent, ims_client, weather_mapper, weather_repository
):
    raw_stations = [make_raw_station(1), make_raw_station(2)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    def map_observation_side_effect(raw):
        if raw["stationId"] == 1:
            raise WeatherMappingError("IMS observation timestamp could not be parsed.")
        return make_domain_observation(raw["stationId"])

    weather_mapper.map_observation.side_effect = map_observation_side_effect

    result = agent.collect()

    # Both stations were still successfully stored; only the observation for
    # station 1 was skipped.
    assert weather_repository.save_station.call_count == 2
    weather_repository.save_observation.assert_called_once()
    assert result.stations_processed == 2
    assert result.stations_failed == 0
    assert result.observations_failed == 1
    assert result.observations_saved == 1


# ---------------------------------------------------------------------------
# Observation persistence failure
# ---------------------------------------------------------------------------


def test_collect_observation_repository_failure_is_isolated_per_station(
    agent, ims_client, weather_mapper, weather_repository
):
    raw_stations = [make_raw_station(1), make_raw_station(2)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    def save_observation_side_effect(observation):
        if observation.station_external_id == 1:
            raise WeatherStationNotStoredError("No stored station with external_station_id=1.")
        return SaveObservationResult(observation=observation, is_duplicate=False)

    weather_repository.save_observation.side_effect = save_observation_side_effect

    result = agent.collect()

    assert result.stations_processed == 2
    assert result.stations_failed == 0
    assert result.observations_failed == 1
    assert result.observations_saved == 1


# ---------------------------------------------------------------------------
# Inactive stations: real IMS /stations includes `active: false` stations
# whose "latest" data can be decades old - they must be skipped entirely.
# ---------------------------------------------------------------------------


def test_collect_inactive_station_is_not_queried_for_observations(
    agent, ims_client, weather_mapper, weather_repository
):
    raw_stations = [
        {**make_raw_station(1), "active": True},
        {**make_raw_station(2), "active": False},
        {**make_raw_station(3), "active": True},
    ]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    result = agent.collect()

    assert ims_client.get_station_data.call_count == 2
    ims_client.get_station_data.assert_any_call(1)
    ims_client.get_station_data.assert_any_call(3)
    assert 2 not in [call.args[0] for call in ims_client.get_station_data.call_args_list]
    # Inactive station is neither mapped nor persisted.
    assert [call.args[0]["stationId"] for call in weather_mapper.map_station.call_args_list] == [1, 3]
    assert weather_repository.save_station.call_count == 2

    assert result.success is True
    assert result.stations_received == 3
    assert result.stations_processed == 2
    assert result.stations_skipped == 1
    assert result.stations_failed == 0
    assert result.observations_saved == 2


def test_collect_active_and_unflagged_stations_are_still_collected(
    agent, ims_client, weather_mapper, weather_repository
):
    # `active: true` and a missing `active` key both keep the existing behavior.
    raw_stations = [{**make_raw_station(1), "active": True}, make_raw_station(2)]
    ims_client.get_stations.return_value = raw_stations
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    result = agent.collect()

    assert ims_client.get_station_data.call_count == 2
    assert result.stations_processed == 2
    assert result.stations_skipped == 0
    assert result.observations_saved == 2


def test_collect_all_inactive_stations_performs_no_observation_requests_or_writes(
    agent, ims_client, weather_mapper, weather_repository
):
    ims_client.get_stations.return_value = [{**make_raw_station(1), "active": False}]
    _wire_happy_path(ims_client, weather_mapper, weather_repository)

    result = agent.collect()

    ims_client.get_station_data.assert_not_called()
    weather_repository.save_station.assert_not_called()
    weather_repository.save_observation.assert_not_called()
    assert result.success is True
    assert result.stations_received == 1
    assert result.stations_skipped == 1
