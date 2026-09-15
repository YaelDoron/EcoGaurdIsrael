"""Tests for the shared spherical-earth geographic utilities."""
from __future__ import annotations

import math

import pytest

from src.utils.geo import EARTH_RADIUS_KM, destination_point, haversine_distance_km, initial_bearing_deg

ONE_DEGREE_KM = EARTH_RADIUS_KM * math.radians(1.0)


# ---------------------------------------------------------------------------
# haversine_distance_km
# ---------------------------------------------------------------------------


def test_haversine_identical_coordinates_is_zero():
    assert haversine_distance_km(32.75, 35.0, 32.75, 35.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_along_equator():
    # On the equator, the haversine formula reduces exactly to R * delta_longitude.
    distance = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
    assert distance == pytest.approx(ONE_DEGREE_KM, rel=1e-9)


def test_haversine_one_degree_along_meridian():
    # Along any meridian, the formula reduces exactly to R * delta_latitude.
    distance = haversine_distance_km(0.0, 35.0, 1.0, 35.0)
    assert distance == pytest.approx(ONE_DEGREE_KM, rel=1e-9)


def test_haversine_known_city_pair():
    # London (51.5007, -0.1246) to Paris (48.8566, 2.3522): well-known ~344 km.
    distance = haversine_distance_km(51.5007, -0.1246, 48.8566, 2.3522)
    assert distance == pytest.approx(344.0, rel=0.01)


def test_haversine_symmetric():
    a = haversine_distance_km(32.75, 35.0, 33.0, 35.5)
    b = haversine_distance_km(33.0, 35.5, 32.75, 35.0)
    assert a == pytest.approx(b, rel=1e-12)


# ---------------------------------------------------------------------------
# initial_bearing_deg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "second_latitude, second_longitude, expected_bearing",
    [
        (1.0, 0.0, 0.0),  # due north
        (0.0, 1.0, 90.0),  # due east
        (-1.0, 0.0, 180.0),  # due south
        (0.0, -1.0, 270.0),  # due west
    ],
)
def test_bearing_cardinal_directions(second_latitude, second_longitude, expected_bearing):
    bearing = initial_bearing_deg(0.0, 0.0, second_latitude, second_longitude)
    assert bearing == pytest.approx(expected_bearing, abs=1e-6)


def test_bearing_is_within_valid_range():
    bearing = initial_bearing_deg(32.75, 35.0, 32.5, 34.7)
    assert 0.0 <= bearing < 360.0


# ---------------------------------------------------------------------------
# destination_point
# ---------------------------------------------------------------------------


def test_destination_point_north_along_meridian():
    latitude, longitude = destination_point(0.0, 0.0, 0.0, ONE_DEGREE_KM)
    assert latitude == pytest.approx(1.0, abs=1e-6)
    assert longitude == pytest.approx(0.0, abs=1e-6)


def test_destination_point_east_along_equator():
    latitude, longitude = destination_point(0.0, 0.0, 90.0, ONE_DEGREE_KM)
    assert latitude == pytest.approx(0.0, abs=1e-6)
    assert longitude == pytest.approx(1.0, abs=1e-6)


def test_destination_point_south_along_meridian():
    latitude, longitude = destination_point(0.0, 0.0, 180.0, ONE_DEGREE_KM)
    assert latitude == pytest.approx(-1.0, abs=1e-6)


def test_destination_point_west_along_equator():
    latitude, longitude = destination_point(0.0, 0.0, 270.0, ONE_DEGREE_KM)
    assert longitude == pytest.approx(-1.0, abs=1e-6)


def test_destination_point_zero_distance_returns_origin():
    latitude, longitude = destination_point(32.75, 35.0, 123.0, 0.0)
    assert latitude == pytest.approx(32.75, abs=1e-9)
    assert longitude == pytest.approx(35.0, abs=1e-9)


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
def test_destination_and_bearing_round_trip(bearing_deg):
    origin_lat, origin_lon = 32.75, 35.0
    distance_km = 3.7
    dest_lat, dest_lon = destination_point(origin_lat, origin_lon, bearing_deg, distance_km)

    assert haversine_distance_km(origin_lat, origin_lon, dest_lat, dest_lon) == pytest.approx(
        distance_km, rel=1e-6
    )
    round_trip_bearing = initial_bearing_deg(origin_lat, origin_lon, dest_lat, dest_lon)
    circular_difference = min(
        abs(round_trip_bearing - bearing_deg % 360.0),
        360.0 - abs(round_trip_bearing - bearing_deg % 360.0),
    )
    assert circular_difference == pytest.approx(0.0, abs=1e-6)
