"""Tests for predefined simulation locations."""

import pytest

from src.simulation import (
    CARMEL_LOCATION,
    GALILEE_LOCATION,
    GOLAN_LOCATION,
    JUDEAN_HILLS_LOCATION,
    JERUSALEM_FOREST_LOCATION,
    SIMULATION_LOCATIONS,
    SimulationLocation,
    get_simulation_location,
)

EXPECTED_LOCATION_KEYS = {
    "carmel",
    "jerusalem_forest",
    "galilee",
    "golan",
    "judean_hills",
}


def test_all_required_locations_exist():
    assert set(SIMULATION_LOCATIONS) == EXPECTED_LOCATION_KEYS
    assert SIMULATION_LOCATIONS["carmel"] == CARMEL_LOCATION
    assert SIMULATION_LOCATIONS["jerusalem_forest"] == JERUSALEM_FOREST_LOCATION
    assert SIMULATION_LOCATIONS["galilee"] == GALILEE_LOCATION
    assert SIMULATION_LOCATIONS["golan"] == GOLAN_LOCATION
    assert SIMULATION_LOCATIONS["judean_hills"] == JUDEAN_HILLS_LOCATION


def test_location_keys_are_stable_and_unique():
    assert len(SIMULATION_LOCATIONS) == len(EXPECTED_LOCATION_KEYS)
    assert all(key == key.lower() for key in SIMULATION_LOCATIONS)
    assert all(" " not in key for key in SIMULATION_LOCATIONS)


def test_predefined_location_coordinates_are_valid():
    for location in SIMULATION_LOCATIONS.values():
        assert isinstance(location, SimulationLocation)
        assert -90 <= location.latitude <= 90
        assert -180 <= location.longitude <= 180


def test_lookup_by_supported_key_returns_location():
    assert get_simulation_location("carmel") == CARMEL_LOCATION
    assert get_simulation_location("golan") == GOLAN_LOCATION


def test_unsupported_location_key_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported simulation location key"):
        get_simulation_location("unknown")


def test_location_registry_is_read_only():
    with pytest.raises(TypeError):
        SIMULATION_LOCATIONS["new_location"] = CARMEL_LOCATION


def test_simulation_location_is_immutable():
    with pytest.raises(Exception):
        CARMEL_LOCATION.latitude = 0
