"""Tests for the shared spherical-earth geographic utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from src.utils.geo import (
    EARTH_RADIUS_KM,
    destination_point,
    haversine_distance_km,
    initial_bearing_deg,
    resolve_nearest_containing_area_name,
)

ONE_DEGREE_KM = EARTH_RADIUS_KM * math.radians(1.0)


@dataclass(frozen=True)
class _FakeArea:
    area_id: str
    area_name: str
    area_latitude: float
    area_longitude: float
    area_radius_km: float


# ---------------------------------------------------------------------------
# resolve_nearest_containing_area_name (shared by satellite hotspots and
# active FireEvents' own location_name enrichment)
# ---------------------------------------------------------------------------


def test_resolve_area_name_point_inside_one_area():
    area = _FakeArea(area_id="area-1", area_name="Carmel Demo Area", area_latitude=32.731, area_longitude=35.046, area_radius_km=5.0)
    result = resolve_nearest_containing_area_name(32.74, 35.05, (area,))
    assert result == "Carmel Demo Area"


def test_resolve_area_name_point_outside_every_area_returns_none():
    area = _FakeArea(area_id="area-1", area_name="Carmel Demo Area", area_latitude=32.731, area_longitude=35.046, area_radius_km=5.0)
    # ~60km away - well outside the 5km radius.
    result = resolve_nearest_containing_area_name(33.2, 35.6, (area,))
    assert result is None


def test_resolve_area_name_no_areas_returns_none():
    assert resolve_nearest_containing_area_name(32.74, 35.05, ()) is None


def test_resolve_area_name_overlapping_areas_pick_nearest_center():
    near = _FakeArea(area_id="area-near", area_name="Near Area", area_latitude=32.740, area_longitude=35.050, area_radius_km=10.0)
    far = _FakeArea(area_id="area-far", area_name="Far Area", area_latitude=32.900, area_longitude=35.200, area_radius_km=30.0)
    # Point sits inside both circles, but is much closer to `near`'s center.
    result = resolve_nearest_containing_area_name(32.741, 35.051, (far, near))
    assert result == "Near Area"


def test_resolve_area_name_exact_distance_tie_breaks_on_area_id():
    # Two areas whose centers are equidistant from the point - "area-a" must
    # win deterministically over "area-b" regardless of input order.
    point_lat, point_lon = 32.75, 35.0
    area_a = _FakeArea(area_id="area-a", area_name="Area A", area_latitude=32.80, area_longitude=35.0, area_radius_km=10.0)
    area_b = _FakeArea(area_id="area-b", area_name="Area B", area_latitude=32.70, area_longitude=35.0, area_radius_km=10.0)

    assert resolve_nearest_containing_area_name(point_lat, point_lon, (area_a, area_b)) == "Area A"
    assert resolve_nearest_containing_area_name(point_lat, point_lon, (area_b, area_a)) == "Area A"


def test_resolve_area_name_being_nearest_is_not_enough_without_containment():
    # The nearest area's circle does NOT contain the point, but a farther
    # area's circle does - the farther-but-containing area must win.
    nearest_but_excludes = _FakeArea(
        area_id="area-near", area_name="Near But Excluded", area_latitude=32.74, area_longitude=35.05, area_radius_km=0.01
    )
    farther_but_contains = _FakeArea(
        area_id="area-far", area_name="Far But Contains", area_latitude=33.0, area_longitude=35.5, area_radius_km=100.0
    )
    result = resolve_nearest_containing_area_name(32.741, 35.051, (nearest_but_excludes, farther_but_contains))
    assert result == "Far But Contains"


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
