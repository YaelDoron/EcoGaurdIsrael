"""Tests for SatelliteDataGenerator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import random

import pytest

from src.models.satellite_hotspot import SatelliteHotspot
from src.simulation import DEFAULT_CARMEL_LOCATION, SIMULATION_LOCATIONS, ScenarioType
from src.simulation.generators.satellite_data_generator import (
    SIMULATED_BRIGHTNESS_RANGE,
    SIMULATED_CONFIDENCE_VALUES,
    SIMULATED_FRP_RANGE,
    SIMULATED_HOTSPOT_MAX_RADIUS_KM,
    SIMULATED_HOTSPOT_MIN_RADIUS_KM,
    SIMULATED_INSTRUMENT,
    SIMULATED_SATELLITE,
    GeneratedSatelliteData,
    SatelliteDataGenerator,
)

TIMESTAMP = datetime(2026, 9, 12, 14, 0, 20, tzinfo=timezone.utc)


def distance_km(first_latitude: float, first_longitude: float, second_latitude: float, second_longitude: float) -> float:
    earth_radius_km = 6371.0
    lat1 = math.radians(first_latitude)
    lon1 = math.radians(first_longitude)
    lat2 = math.radians(second_latitude)
    lon2 = math.radians(second_longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def assert_hotspot_valid_for_location(hotspot: SatelliteHotspot, location, timestamp: datetime) -> None:
    assert isinstance(hotspot, SatelliteHotspot)
    assert -90 <= hotspot.latitude <= 90
    assert -180 <= hotspot.longitude <= 180
    assert hotspot.detected_at is timestamp
    assert SIMULATED_FRP_RANGE[0] <= hotspot.frp <= SIMULATED_FRP_RANGE[1]
    assert SIMULATED_BRIGHTNESS_RANGE[0] <= hotspot.brightness <= SIMULATED_BRIGHTNESS_RANGE[1]
    assert hotspot.confidence in SIMULATED_CONFIDENCE_VALUES
    assert hotspot.satellite == SIMULATED_SATELLITE
    assert hotspot.instrument == SIMULATED_INSTRUMENT
    assert hotspot.day_night in {"D", "N"}
    hotspot_distance = distance_km(location.latitude, location.longitude, hotspot.latitude, hotspot.longitude)
    assert SIMULATED_HOTSPOT_MIN_RADIUS_KM <= hotspot_distance <= SIMULATED_HOTSPOT_MAX_RADIUS_KM


@pytest.mark.parametrize("scenario_type", [ScenarioType.LOW_RISK_NO_FIRE, ScenarioType.HIGH_RISK_NO_FIRE])
def test_no_fire_scenarios_generate_no_hotspots_for_every_location(scenario_type):
    generator = SatelliteDataGenerator(seed=42)

    for location in SIMULATION_LOCATIONS.values():
        generated = generator.generate(
            scenario_type=scenario_type,
            timestamp=TIMESTAMP,
            location=location,
        )

        assert generated == GeneratedSatelliteData(hotspots=())


def test_active_fire_generates_valid_satellite_hotspot():
    generated = SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
    )

    assert isinstance(generated, GeneratedSatelliteData)
    assert len(generated.hotspots) >= 1
    assert_hotspot_valid_for_location(generated.hotspots[0], DEFAULT_CARMEL_LOCATION, TIMESTAMP)
    assert not hasattr(generated, "fire_confirmed")
    assert not hasattr(generated, "risk_score")
    assert not hasattr(generated.hotspots[0], "fire_probability")


def test_generated_hotspot_carries_the_canonical_location_name():
    """Part G/H: the simulation already knows the canonical location - the
    generated hotspot must carry it forward as trustworthy provenance for
    FireEvent creation, never left for a later read-side guess."""
    generated = SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
    )

    assert all(hotspot.location_name == DEFAULT_CARMEL_LOCATION.name for hotspot in generated.hotspots)


def test_same_seed_and_inputs_generate_same_hotspots_across_instances():
    first = SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
    )
    second = SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
    )

    assert first == second


def test_different_satellite_event_timestamps_vary_but_remain_local():
    generator = SatelliteDataGenerator(seed=42)
    first = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
    )
    second_timestamp = TIMESTAMP + timedelta(seconds=60)
    second = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=second_timestamp,
        location=DEFAULT_CARMEL_LOCATION,
    )

    assert first.hotspots != second.hotspots
    assert_hotspot_valid_for_location(first.hotspots[0], DEFAULT_CARMEL_LOCATION, TIMESTAMP)
    assert_hotspot_valid_for_location(second.hotspots[0], DEFAULT_CARMEL_LOCATION, second_timestamp)


def test_generator_does_not_mutate_global_random_sequence():
    random.seed(12345)
    expected_first = random.random()
    expected_second = random.random()

    random.seed(12345)
    actual_first = random.random()
    SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TIMESTAMP,
        location=DEFAULT_CARMEL_LOCATION,
    )
    actual_second = random.random()

    assert actual_first == expected_first
    assert actual_second == expected_second


def test_active_fire_hotspot_is_near_each_predefined_location_and_not_other_anchors():
    generator = SatelliteDataGenerator(seed=42)

    for key, location in SIMULATION_LOCATIONS.items():
        generated = generator.generate(
            scenario_type=ScenarioType.ACTIVE_FIRE,
            timestamp=TIMESTAMP,
            location=location,
        )
        hotspot = generated.hotspots[0]
        assert_hotspot_valid_for_location(hotspot, location, TIMESTAMP)

        own_distance = distance_km(location.latitude, location.longitude, hotspot.latitude, hotspot.longitude)
        for other_key, other_location in SIMULATION_LOCATIONS.items():
            if other_key == key:
                continue
            other_distance = distance_km(
                other_location.latitude,
                other_location.longitude,
                hotspot.latitude,
                hotspot.longitude,
            )
            assert own_distance < other_distance


def test_day_night_is_derived_from_timestamp_hour():
    day_hotspot = SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc),
        location=DEFAULT_CARMEL_LOCATION,
    ).hotspots[0]
    night_hotspot = SatelliteDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=datetime(2026, 9, 12, 22, 0, tzinfo=timezone.utc),
        location=DEFAULT_CARMEL_LOCATION,
    ).hotspots[0]

    assert day_hotspot.day_night == "D"
    assert night_hotspot.day_night == "N"


def test_invalid_inputs_are_rejected():
    generator = SatelliteDataGenerator(seed=42)

    with pytest.raises(ValueError):
        SatelliteDataGenerator(seed=True)
    with pytest.raises(ValueError):
        generator.generate("active_fire", TIMESTAMP, DEFAULT_CARMEL_LOCATION)
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, "2026-09-12", DEFAULT_CARMEL_LOCATION)
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location="Carmel")
