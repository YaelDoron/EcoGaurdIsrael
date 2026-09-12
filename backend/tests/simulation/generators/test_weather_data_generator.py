"""Tests for WeatherDataGenerator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import random

import pytest

from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.simulation import DEFAULT_CARMEL_LOCATION, SIMULATION_LOCATIONS, ScenarioType, SimulationLocation
from src.simulation.generators.weather_data_generator import (
    SIMULATED_WEATHER_STATION_ID_BASE,
    SIMULATED_WEATHER_STATION_ID_SPAN,
    SIMULATED_WEATHER_STATIONS_PER_LOCATION,
    WEATHER_SCENARIO_PROFILES,
    GeneratedStationWeather,
    GeneratedWeatherData,
    WeatherDataGenerator,
)

TIMESTAMP = datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)


def assert_observation_in_profile(observation: WeatherObservation, scenario_type: ScenarioType) -> None:
    profile = WEATHER_SCENARIO_PROFILES[scenario_type]
    assert profile.temperature_celsius[0] <= observation.temperature <= profile.temperature_celsius[1]
    assert (
        profile.relative_humidity_percent[0]
        <= observation.relative_humidity
        <= profile.relative_humidity_percent[1]
    )
    assert profile.wind_speed_kmh[0] <= observation.wind_speed <= profile.wind_speed_kmh[1]
    assert observation.wind_gust >= observation.wind_speed
    assert 0 <= observation.wind_direction <= 360
    assert profile.rainfall_mm[0] <= observation.rainfall <= profile.rainfall_mm[1]


def assert_generated_data_valid(generated: GeneratedWeatherData, scenario_type: ScenarioType, timestamp: datetime) -> None:
    assert len(generated.measurements) == SIMULATED_WEATHER_STATIONS_PER_LOCATION
    assert len(generated.stations) == SIMULATED_WEATHER_STATIONS_PER_LOCATION
    assert len(generated.observations) == SIMULATED_WEATHER_STATIONS_PER_LOCATION
    for measurement in generated.measurements:
        assert isinstance(measurement, GeneratedStationWeather)
        assert isinstance(measurement.station, WeatherStation)
        assert isinstance(measurement.observation, WeatherObservation)
        assert measurement.observation.station_external_id == measurement.station.external_station_id
        assert measurement.observation.timestamp is timestamp
        assert_observation_in_profile(measurement.observation, scenario_type)


@pytest.mark.parametrize("scenario_type", list(ScenarioType))
def test_generated_data_uses_existing_weather_domain_models(scenario_type):
    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=scenario_type,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    assert isinstance(generated, GeneratedWeatherData)
    assert_generated_data_valid(generated, scenario_type, TIMESTAMP)


@pytest.mark.parametrize("scenario_type", list(ScenarioType))
def test_many_samples_stay_inside_configured_profile(scenario_type):
    generator = WeatherDataGenerator(seed=42)

    for minute in range(60):
        timestamp = TIMESTAMP + timedelta(minutes=minute)
        generated = generator.generate(
            scenario_type=scenario_type,
            location=DEFAULT_CARMEL_LOCATION,
            timestamp=timestamp,
        )
        assert_generated_data_valid(generated, scenario_type, timestamp)


def test_low_risk_allows_non_extreme_conditions():
    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.LOW_RISK_NO_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    assert_generated_data_valid(generated, ScenarioType.LOW_RISK_NO_FIRE, TIMESTAMP)


def test_high_risk_has_hot_dry_no_rain_profile_without_fire_output():
    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.HIGH_RISK_NO_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    assert_generated_data_valid(generated, ScenarioType.HIGH_RISK_NO_FIRE, TIMESTAMP)
    for observation in generated.observations:
        assert observation.temperature >= 34
        assert observation.relative_humidity <= 25
        assert observation.rainfall == 0
    assert not hasattr(generated, "fire_detected")
    assert not hasattr(generated, "risk_score")


def test_active_fire_weather_profile_has_no_fire_analysis_output():
    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    assert_generated_data_valid(generated, ScenarioType.ACTIVE_FIRE, TIMESTAMP)
    assert all(observation.rainfall == 0 for observation in generated.observations)
    assert not hasattr(generated, "fire_detected")
    assert not hasattr(generated, "fire_probability")


def test_same_seed_and_inputs_generate_same_data_across_instances():
    first = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )
    second = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    assert first == second


def test_different_weather_event_timestamps_vary_measurements_but_keep_same_station():
    generator = WeatherDataGenerator(seed=42)

    first = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )
    second = generator.generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP + timedelta(seconds=65),
    )

    assert first.stations == second.stations
    assert first.observations != second.observations
    assert all(observation.timestamp == TIMESTAMP for observation in first.observations)
    assert all(
        observation.timestamp == TIMESTAMP + timedelta(seconds=65)
        for observation in second.observations
    )
    for observation in first.observations:
        assert_observation_in_profile(observation, ScenarioType.ACTIVE_FIRE)
    for observation in second.observations:
        assert_observation_in_profile(observation, ScenarioType.ACTIVE_FIRE)


def test_generator_does_not_mutate_global_random_sequence():
    random.seed(12345)
    expected_first = random.random()
    expected_second = random.random()

    random.seed(12345)
    actual_first = random.random()
    WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )
    actual_second = random.random()

    assert actual_first == expected_first
    assert actual_second == expected_second


def test_station_identity_is_simulated_deterministic_and_stable_across_observations():
    location = SimulationLocation(name="Carmel Ridge", latitude=32.73, longitude=35.04)
    generator = WeatherDataGenerator(seed=1)

    first = generator.generate(
        scenario_type=ScenarioType.LOW_RISK_NO_FIRE,
        location=location,
        timestamp=TIMESTAMP,
    )
    second = generator.generate(
        scenario_type=ScenarioType.HIGH_RISK_NO_FIRE,
        location=location,
        timestamp=TIMESTAMP + timedelta(minutes=5),
    )

    assert first.stations == second.stations
    for measurement in first.measurements:
        assert measurement.station.external_station_id >= SIMULATED_WEATHER_STATION_ID_BASE
        assert measurement.station.external_station_id < (
            SIMULATED_WEATHER_STATION_ID_BASE + SIMULATED_WEATHER_STATION_ID_SPAN
        )
        assert measurement.station.name.startswith("SIM-CARMEL-RIDGE-")
        assert measurement.observation.station_external_id == measurement.station.external_station_id


def test_exact_timezone_aware_timestamp_is_preserved():
    timestamp = datetime(2026, 9, 12, 17, 1, 5, tzinfo=timezone.utc)

    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=timestamp,
    )

    assert all(observation.timestamp is timestamp for observation in generated.observations)


def test_every_predefined_location_gets_three_unique_nearby_stations():
    generator = WeatherDataGenerator(seed=42)

    for key, location in SIMULATION_LOCATIONS.items():
        generated = generator.generate(
            scenario_type=ScenarioType.ACTIVE_FIRE,
            location=location,
            timestamp=TIMESTAMP,
        )
        station_ids = [station.external_station_id for station in generated.stations]
        station_names = [station.name for station in generated.stations]
        coordinates = [(station.latitude, station.longitude) for station in generated.stations]

        assert len(generated.stations) == 3
        assert len(set(station_ids)) == 3
        assert len(set(station_names)) == 3
        assert len(set(coordinates)) == 3
        assert all(station.name.startswith(f"SIM-{key.replace('_', '-').upper()}-") for station in generated.stations)
        for station in generated.stations:
            assert 0 < abs(station.latitude - location.latitude) < 0.02
            assert 0 <= abs(station.longitude - location.longitude) < 0.02
            assert (station.latitude, station.longitude) != (location.latitude, location.longitude)


def test_station_network_is_same_across_scenario_types_for_same_location():
    generator = WeatherDataGenerator(seed=42)
    networks = [
        generator.generate(scenario_type=scenario_type, location=DEFAULT_CARMEL_LOCATION, timestamp=TIMESTAMP).stations
        for scenario_type in ScenarioType
    ]

    assert networks[0] == networks[1] == networks[2]


def test_different_stations_receive_deterministic_variation():
    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    observations_without_station_id = {
        (
            observation.temperature,
            observation.relative_humidity,
            observation.wind_speed,
            observation.wind_direction,
            observation.wind_gust,
            observation.rainfall,
        )
        for observation in generated.observations
    }
    assert len(observations_without_station_id) > 1


def test_invalid_inputs_are_rejected():
    generator = WeatherDataGenerator(seed=42)

    with pytest.raises(ValueError):
        WeatherDataGenerator(seed=True)
    with pytest.raises(ValueError):
        generator.generate("active_fire", DEFAULT_CARMEL_LOCATION, TIMESTAMP)
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, DEFAULT_CARMEL_LOCATION, "2026-09-12")
    with pytest.raises(ValueError):
        generator.generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location="Carmel")
