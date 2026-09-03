"""Unit tests for WeatherMapper.

All inputs are deterministic dictionaries - no IMS/network/database access,
no API token, and no reliance on system time.
"""
from datetime import datetime

import pytest

from src.mappers.exceptions import MissingRequiredWeatherFieldError, WeatherMappingError
from src.mappers.weather_mapper import WeatherMapper
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

RAW_STATION = {
    "stationId": 17,
    "name": "HAIFA",
    "active": True,
    "location": {
        "latitude": 32.79,
        "longitude": 34.99,
    },
    "regionId": 3,
    "monitors": [],
}

RAW_OBSERVATION = {
    "stationId": 17,
    "datetime": "2026-09-02T12:30:00",
    "channels": [
        {"name": "TD", "value": 31.4, "valid": True},
        {"name": "RH", "value": 42, "valid": True},
        {"name": "WS", "value": 5.8, "valid": True},
        {"name": "WD", "value": 240, "valid": True},
        {"name": "WSmax", "value": 8.2, "valid": True},
        {"name": "Rain", "value": 0, "valid": True},
    ],
}


def _channels(*entries):
    return {"stationId": 17, "datetime": "2026-09-02T12:30:00", "channels": list(entries)}


# ---------------------------------------------------------------------------
# Station mapping
# ---------------------------------------------------------------------------


def test_map_station_valid_raw_station_produces_correct_weather_station():
    station = WeatherMapper.map_station(RAW_STATION)

    assert station == WeatherStation(
        external_station_id=17,
        name="HAIFA",
        latitude=32.79,
        longitude=34.99,
        region_id=3,
        active=True,
    )


def test_map_station_missing_region_id_becomes_none():
    raw = dict(RAW_STATION)
    del raw["regionId"]

    station = WeatherMapper.map_station(raw)

    assert station.region_id is None


def test_map_station_missing_active_becomes_none():
    raw = dict(RAW_STATION)
    del raw["active"]

    station = WeatherMapper.map_station(raw)

    assert station.active is None


def test_map_station_missing_station_id_raises_mapping_error():
    raw = dict(RAW_STATION)
    del raw["stationId"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_missing_name_raises_mapping_error():
    raw = dict(RAW_STATION)
    del raw["name"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_missing_latitude_raises_mapping_error():
    raw = {**RAW_STATION, "location": {"longitude": 34.99}}

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_missing_longitude_raises_mapping_error():
    raw = {**RAW_STATION, "location": {"latitude": 32.79}}

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_invalid_coordinates_raise_mapping_error():
    raw = {**RAW_STATION, "location": {"latitude": 999, "longitude": 34.99}}

    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_station(raw)


def test_map_stations_maps_a_list_of_raw_stations():
    raw_list = [RAW_STATION, {**RAW_STATION, "stationId": 21, "name": "SECOND"}]

    stations = WeatherMapper.map_stations(raw_list)

    assert [s.external_station_id for s in stations] == [17, 21]


# ---------------------------------------------------------------------------
# Observation mapping
# ---------------------------------------------------------------------------


def test_map_observation_valid_raw_observation_maps_all_six_channels():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert observation.station_external_id == 17
    assert observation.temperature == 31.4
    assert observation.relative_humidity == 42
    assert observation.wind_speed == 5.8
    assert observation.wind_direction == 240
    assert observation.wind_gust == 8.2
    assert observation.rainfall == 0


def test_map_observation_missing_wsmax_channel_becomes_none():
    raw = _channels({"name": "TD", "value": 31.4, "valid": True})

    observation = WeatherMapper.map_observation(raw)

    assert observation.wind_gust is None


def test_map_observation_wsmax_invalid_becomes_none():
    raw = _channels({"name": "WSmax", "value": 8.2, "valid": False})

    observation = WeatherMapper.map_observation(raw)

    assert observation.wind_gust is None


def test_map_observation_td_invalid_becomes_none_even_with_value_present():
    raw = _channels({"name": "TD", "value": 35.2, "valid": False})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None


def test_map_observation_unknown_channel_is_ignored():
    raw = _channels(
        {"name": "TD", "value": 31.4, "valid": True},
        {"name": "Pressure", "value": 1013, "valid": True},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 31.4
    # No unexpected attribute should exist for the unknown channel.
    assert not hasattr(observation, "pressure")


def test_map_observation_missing_station_id_raises_mapping_error():
    raw = dict(RAW_OBSERVATION)
    del raw["stationId"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation(raw)


def test_map_observation_missing_datetime_raises_mapping_error():
    raw = dict(RAW_OBSERVATION)
    del raw["datetime"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation(raw)


def test_map_observation_invalid_datetime_raises_mapping_error():
    raw = {**RAW_OBSERVATION, "datetime": "not-a-timestamp"}

    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_observation(raw)


def test_map_observation_numeric_string_values_are_converted():
    raw = _channels(
        {"name": "TD", "value": "31.4", "valid": True},
        {"name": "RH", "value": "42", "valid": True},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 31.4
    assert observation.relative_humidity == 42.0


def test_map_observation_invalid_numeric_value_becomes_none():
    raw = _channels({"name": "TD", "value": "abc", "valid": True})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None


def test_map_observation_missing_channels_key_does_not_fail():
    raw = {"stationId": 17, "datetime": "2026-09-02T12:30:00"}

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None
    assert observation.relative_humidity is None
    assert observation.wind_speed is None
    assert observation.wind_direction is None
    assert observation.wind_gust is None
    assert observation.rainfall is None


def test_map_observation_timestamp_type_is_datetime():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert isinstance(observation.timestamp, datetime)
    assert observation.timestamp == datetime(2026, 9, 2, 12, 30, 0)


def test_map_observation_duplicate_channel_uses_last_valid_occurrence():
    raw = _channels(
        {"name": "TD", "value": 20.0, "valid": True},
        {"name": "TD", "value": 25.0, "valid": True},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 25.0


def test_map_observation_duplicate_channel_later_invalid_keeps_earlier_valid_value():
    raw = _channels(
        {"name": "TD", "value": 20.0, "valid": True},
        {"name": "TD", "value": 25.0, "valid": False},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 20.0


def test_map_observation_no_valid_occurrence_of_duplicate_channel_is_none():
    raw = _channels(
        {"name": "TD", "value": 20.0, "valid": False},
        {"name": "TD", "value": 25.0, "valid": False},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None


def test_map_observation_missing_valid_key_with_value_is_treated_as_valid():
    raw = _channels({"name": "TD", "value": 31.4})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 31.4


def test_map_observation_returns_weather_observation_instance():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert isinstance(observation, WeatherObservation)
