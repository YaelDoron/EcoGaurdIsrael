"""Unit tests for the WeatherStation model."""
import pytest

from src.models.weather_station import WeatherStation


def make_station(**overrides):
    defaults = dict(
        external_station_id=17,
        name="HAIFA",
        latitude=32.79,
        longitude=34.99,
        region_id=3,
        active=True,
    )
    defaults.update(overrides)
    return WeatherStation(**defaults)


def test_valid_station_is_created_successfully():
    station = make_station()

    assert station.external_station_id == 17
    assert station.name == "HAIFA"
    assert station.latitude == 32.79
    assert station.longitude == 34.99
    assert station.region_id == 3
    assert station.active is True


@pytest.mark.parametrize("external_station_id", [0, -1, "abc", None, 1.5, True])
def test_invalid_external_station_id_raises_value_error(external_station_id):
    with pytest.raises(ValueError):
        make_station(external_station_id=external_station_id)


@pytest.mark.parametrize("name", ["", "   ", None, 123])
def test_invalid_name_raises_value_error(name):
    with pytest.raises(ValueError):
        make_station(name=name)


@pytest.mark.parametrize("latitude", [-90.1, 90.1, -1000, "abc", None])
def test_invalid_latitude_raises_value_error(latitude):
    with pytest.raises(ValueError):
        make_station(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, 1000, "abc", None])
def test_invalid_longitude_raises_value_error(longitude):
    with pytest.raises(ValueError):
        make_station(longitude=longitude)


def test_region_id_none_is_allowed():
    station = make_station(region_id=None)

    assert station.region_id is None


@pytest.mark.parametrize("region_id", [0, -1, "abc"])
def test_invalid_region_id_raises_value_error(region_id):
    with pytest.raises(ValueError):
        make_station(region_id=region_id)


def test_active_none_is_allowed():
    station = make_station(active=None)

    assert station.active is None


def test_invalid_active_raises_value_error():
    with pytest.raises(ValueError):
        make_station(active="yes")


def test_boundary_coordinates_are_allowed():
    station = make_station(latitude=-90, longitude=180)

    assert station.latitude == -90
    assert station.longitude == 180
