"""Tests for the pure wildfire-spread calculator input model."""
from __future__ import annotations

import math

import pytest

from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput


def make_input(**overrides) -> FireSpreadInput:
    defaults = dict(
        origin_latitude=32.75,
        origin_longitude=35.0,
        wind_speed_kmh=20.0,
        wind_direction_deg=270.0,
        fuel_moisture_percent=12.0,
        fuel_class=FireSpreadFuelClass.SHRUBS,
        horizon_minutes=30,
    )
    defaults.update(overrides)
    return FireSpreadInput(**defaults)


def test_valid_input_construction():
    input_data = make_input()

    assert input_data.fuel_class is FireSpreadFuelClass.SHRUBS
    assert input_data.horizon_minutes == 30


@pytest.mark.parametrize("horizon_minutes", [30, 60])
def test_supported_horizons_accepted(horizon_minutes):
    input_data = make_input(horizon_minutes=horizon_minutes)
    assert input_data.horizon_minutes == horizon_minutes


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf])
def test_invalid_origin_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_input(origin_latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf])
def test_invalid_origin_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_input(origin_longitude=longitude)


@pytest.mark.parametrize("wind_speed_kmh", [-0.1, -100.0, math.nan, math.inf])
def test_invalid_wind_speed_rejected(wind_speed_kmh):
    with pytest.raises(ValueError):
        make_input(wind_speed_kmh=wind_speed_kmh)


def test_zero_wind_speed_accepted():
    input_data = make_input(wind_speed_kmh=0.0)
    assert input_data.wind_speed_kmh == 0.0


@pytest.mark.parametrize("wind_direction_deg", [-0.1, 360.1, math.nan, math.inf, -math.inf])
def test_invalid_wind_direction_rejected(wind_direction_deg):
    with pytest.raises(ValueError):
        make_input(wind_direction_deg=wind_direction_deg)


@pytest.mark.parametrize("wind_direction_deg", [0.0, 360.0])
def test_wind_direction_boundary_values_accepted(wind_direction_deg):
    input_data = make_input(wind_direction_deg=wind_direction_deg)
    assert input_data.wind_direction_deg == wind_direction_deg


@pytest.mark.parametrize("fuel_moisture_percent", [-0.1, -50.0, math.nan, math.inf])
def test_invalid_fuel_moisture_rejected(fuel_moisture_percent):
    with pytest.raises(ValueError):
        make_input(fuel_moisture_percent=fuel_moisture_percent)


def test_zero_fuel_moisture_accepted():
    input_data = make_input(fuel_moisture_percent=0.0)
    assert input_data.fuel_moisture_percent == 0.0


def test_invalid_fuel_class_rejected():
    with pytest.raises(ValueError):
        make_input(fuel_class="shrubs")


@pytest.mark.parametrize("horizon_minutes", [0, 15, 45, 90, -30, True])
def test_invalid_horizon_rejected(horizon_minutes):
    with pytest.raises(ValueError):
        make_input(horizon_minutes=horizon_minutes)


@pytest.mark.parametrize("field", ["origin_latitude", "origin_longitude", "wind_speed_kmh", "fuel_moisture_percent"])
def test_bool_rejected_for_numeric_fields(field):
    with pytest.raises(ValueError):
        make_input(**{field: True})


def test_wind_direction_bool_rejected():
    with pytest.raises(ValueError):
        make_input(wind_direction_deg=True)
