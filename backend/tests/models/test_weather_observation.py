"""Unit tests for the WeatherObservation model."""
from datetime import datetime

import pytest

from src.models.weather_observation import WeatherObservation

FIXED_TIMESTAMP = datetime(2026, 9, 2, 12, 30, 0)


def make_observation(**overrides):
    defaults = dict(
        station_external_id=17,
        timestamp=FIXED_TIMESTAMP,
        temperature=31.4,
        relative_humidity=42,
        wind_speed=5.8,
        wind_direction=240,
        wind_gust=8.2,
        rainfall=0,
    )
    defaults.update(overrides)
    return WeatherObservation(**defaults)


def test_valid_observation_is_created_successfully():
    observation = make_observation()

    assert observation.station_external_id == 17
    assert observation.timestamp == FIXED_TIMESTAMP
    assert observation.temperature == 31.4
    assert observation.relative_humidity == 42
    assert observation.wind_speed == 5.8
    assert observation.wind_direction == 240
    assert observation.wind_gust == 8.2
    assert observation.rainfall == 0


def test_observation_with_all_optional_measurements_none_is_allowed():
    observation = make_observation(
        temperature=None,
        relative_humidity=None,
        wind_speed=None,
        wind_direction=None,
        wind_gust=None,
        rainfall=None,
    )

    assert observation.temperature is None
    assert observation.relative_humidity is None
    assert observation.wind_speed is None
    assert observation.wind_direction is None
    assert observation.wind_gust is None
    assert observation.rainfall is None


@pytest.mark.parametrize("station_external_id", [0, -1, "abc", None, True])
def test_invalid_station_external_id_raises_value_error(station_external_id):
    with pytest.raises(ValueError):
        make_observation(station_external_id=station_external_id)


@pytest.mark.parametrize("timestamp", ["2026-09-02T12:30:00", None, 12345])
def test_non_datetime_timestamp_raises_value_error(timestamp):
    with pytest.raises(ValueError):
        make_observation(timestamp=timestamp)


def test_humidity_below_zero_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(relative_humidity=-1)


def test_humidity_above_100_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(relative_humidity=101)


def test_negative_wind_speed_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(wind_speed=-0.1)


def test_wind_direction_below_zero_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(wind_direction=-1)


def test_wind_direction_above_360_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(wind_direction=361)


def test_negative_rainfall_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(rainfall=-0.1)


def test_negative_wind_gust_raises_value_error():
    with pytest.raises(ValueError):
        make_observation(wind_gust=-0.1)


def test_boundary_humidity_and_wind_direction_are_allowed():
    observation = make_observation(relative_humidity=0, wind_direction=360)

    assert observation.relative_humidity == 0
    assert observation.wind_direction == 360
