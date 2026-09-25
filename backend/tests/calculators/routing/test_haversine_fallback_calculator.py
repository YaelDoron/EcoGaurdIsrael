"""Tests for the straight-line (Haversine) off-road fallback estimate."""
from __future__ import annotations

import pytest

from src.calculators.routing.haversine_fallback_calculator import (
    DEFAULT_OFF_ROAD_SPEED_KMH,
    DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH,
    MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM,
    estimate_disconnected_graph_distance_and_eta,
    estimate_off_road_distance_and_eta,
    is_degenerate_zero_result,
)


def test_is_degenerate_zero_result_true_when_either_field_is_zero():
    assert is_degenerate_zero_result(0.0, 0.0) is True
    assert is_degenerate_zero_result(0.0, 120.0) is True
    assert is_degenerate_zero_result(800.0, 0.0) is True


def test_is_degenerate_zero_result_false_for_a_real_nonzero_route():
    assert is_degenerate_zero_result(800.0, 120.0) is False


def test_estimate_off_road_distance_and_eta_uses_the_default_speed():
    # Two points exactly 1 degree of latitude apart (~111.19 km, per this
    # module's shared EARTH_RADIUS_KM), at the default 40 km/h.
    distance_meters, travel_time_seconds = estimate_off_road_distance_and_eta(32.0, 35.0, 33.0, 35.0)

    assert distance_meters == pytest.approx(111_194.9, rel=1e-3)
    expected_hours = (distance_meters / 1000.0) / DEFAULT_OFF_ROAD_SPEED_KMH
    assert travel_time_seconds == pytest.approx(expected_hours * 3600.0)


def test_estimate_off_road_distance_and_eta_honors_a_custom_speed():
    fast = estimate_off_road_distance_and_eta(32.0, 35.0, 32.1, 35.0, speed_kmh=80.0)
    slow = estimate_off_road_distance_and_eta(32.0, 35.0, 32.1, 35.0, speed_kmh=40.0)

    # Same distance, half the speed -> double the ETA.
    assert fast[0] == pytest.approx(slow[0])
    assert slow[1] == pytest.approx(fast[1] * 2, rel=1e-6)


def test_estimate_off_road_distance_and_eta_is_zero_for_identical_coordinates():
    distance_meters, travel_time_seconds = estimate_off_road_distance_and_eta(32.7, 35.0, 32.7, 35.0)

    assert distance_meters == pytest.approx(0.0)
    assert travel_time_seconds == pytest.approx(0.0)


def test_disconnected_graph_fallback_speed_is_30_kmh_and_slower_than_the_default():
    assert DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH == 30.0
    assert DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH < DEFAULT_OFF_ROAD_SPEED_KMH


def test_disconnected_graph_estimate_is_haversine_distance_at_30_kmh():
    # 0.3 degrees of latitude ~ 33.36 km (inside the cap) -> ~1.112 h at 30 km/h.
    distance_meters, travel_time_seconds = estimate_disconnected_graph_distance_and_eta(32.0, 35.0, 32.3, 35.0)

    assert distance_meters == pytest.approx(33_358.5, rel=1e-3)
    assert travel_time_seconds == pytest.approx((distance_meters / 1000.0) / 30.0 * 3600.0)


def test_disconnected_graph_estimate_for_identical_points_is_zero():
    assert estimate_disconnected_graph_distance_and_eta(32.0, 35.0, 32.0, 35.0) == (0.0, 0.0)


def test_disconnected_graph_estimate_is_none_beyond_the_max_distance():
    assert MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM == 50.0
    # 1 degree of latitude ~ 111.19 km: well past the cap.
    assert estimate_disconnected_graph_distance_and_eta(32.0, 35.0, 33.0, 35.0) is None


def test_disconnected_graph_estimate_applies_just_inside_the_max_distance():
    # 0.44 degrees of latitude ~ 48.9 km: just inside the cap.
    assert estimate_disconnected_graph_distance_and_eta(32.0, 35.0, 32.44, 35.0) is not None
