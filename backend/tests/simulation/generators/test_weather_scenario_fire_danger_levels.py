"""Verify simulated no-fire weather profiles classify correctly against the
real FFWI methodology, not just against their own configured ranges.

These tests intentionally do not assume that a profile named LOW/HIGH still
produces that FireDangerLevel after unrelated changes to either the profile
ranges or the FFWI thresholds -- they run generated weather through the
actual FireDangerInput/FFWICalculator pipeline used in production.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import pytest

from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.models.fire_danger_input import FireDangerInput
from src.models.fire_danger_level import FireDangerLevel
from src.simulation import DEFAULT_CARMEL_LOCATION, ScenarioType
from src.simulation.generators.weather_data_generator import (
    WEATHER_SCENARIO_PROFILES,
    WeatherDataGenerator,
)

TIMESTAMP = datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)
CALCULATOR = FFWICalculator()


def _classify(temperature_c: float, relative_humidity_pct: float, wind_speed_kmh: float) -> FireDangerLevel:
    return CALCULATOR.calculate(
        FireDangerInput(
            temperature_c=temperature_c,
            relative_humidity_pct=relative_humidity_pct,
            wind_speed_kmh=wind_speed_kmh,
        )
    ).level


def _profile_corner_levels(scenario_type: ScenarioType) -> set[FireDangerLevel]:
    """Classify every corner of a profile's weather box (its worst cases)."""
    profile = WEATHER_SCENARIO_PROFILES[scenario_type]
    levels = set()
    for temperature_c, relative_humidity_pct, wind_speed_kmh in product(
        profile.temperature_celsius,
        profile.relative_humidity_percent,
        profile.wind_speed_kmh,
    ):
        levels.add(_classify(temperature_c, relative_humidity_pct, wind_speed_kmh))
    return levels


def _generated_average_level(scenario_type: ScenarioType, seed: int) -> FireDangerLevel:
    """Reproduce FireDangerInputService's cross-station mean for a generated sample."""
    generated = WeatherDataGenerator(seed=seed).generate(
        scenario_type=scenario_type,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )
    observations = generated.observations
    temperature_c = sum(observation.temperature for observation in observations) / len(observations)
    relative_humidity_pct = sum(observation.relative_humidity for observation in observations) / len(observations)
    wind_speed_kmh = sum(observation.wind_speed for observation in observations) / len(observations)
    return _classify(temperature_c, relative_humidity_pct, wind_speed_kmh)


def test_low_risk_no_fire_worst_case_corners_classify_as_low():
    levels = _profile_corner_levels(ScenarioType.LOW_RISK_NO_FIRE)
    assert levels == {FireDangerLevel.LOW}


def test_moderate_risk_no_fire_worst_case_corners_classify_as_moderate():
    levels = _profile_corner_levels(ScenarioType.MODERATE_RISK_NO_FIRE)
    assert levels == {FireDangerLevel.MODERATE}


def test_high_risk_no_fire_worst_case_corners_classify_as_high_or_higher():
    """HIGH_RISK_NO_FIRE is documented to span HIGH..EXTREME by design, never below HIGH."""
    levels = _profile_corner_levels(ScenarioType.HIGH_RISK_NO_FIRE)
    assert levels <= {FireDangerLevel.HIGH, FireDangerLevel.VERY_HIGH, FireDangerLevel.EXTREME}
    assert FireDangerLevel.LOW not in levels
    assert FireDangerLevel.MODERATE not in levels


@pytest.mark.parametrize("seed", [1, 42, 99, 12345])
def test_generated_low_risk_weather_averages_to_low(seed):
    assert _generated_average_level(ScenarioType.LOW_RISK_NO_FIRE, seed) is FireDangerLevel.LOW


@pytest.mark.parametrize("seed", [1, 42, 99, 12345])
def test_generated_moderate_risk_weather_averages_to_moderate(seed):
    assert _generated_average_level(ScenarioType.MODERATE_RISK_NO_FIRE, seed) is FireDangerLevel.MODERATE


@pytest.mark.parametrize("seed", [1, 42, 99, 12345])
def test_generated_high_risk_weather_averages_to_high_or_higher(seed):
    level = _generated_average_level(ScenarioType.HIGH_RISK_NO_FIRE, seed)
    assert level in {FireDangerLevel.HIGH, FireDangerLevel.VERY_HIGH, FireDangerLevel.EXTREME}


def test_moderate_risk_weather_contains_all_mandatory_fire_danger_fields():
    generated = WeatherDataGenerator(seed=42).generate(
        scenario_type=ScenarioType.MODERATE_RISK_NO_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )
    for observation in generated.observations:
        assert observation.temperature is not None
        assert observation.relative_humidity is not None
        assert observation.wind_speed is not None


def test_moderate_risk_weather_is_deterministic_for_same_seed():
    first = WeatherDataGenerator(seed=7).generate(
        scenario_type=ScenarioType.MODERATE_RISK_NO_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )
    second = WeatherDataGenerator(seed=7).generate(
        scenario_type=ScenarioType.MODERATE_RISK_NO_FIRE,
        location=DEFAULT_CARMEL_LOCATION,
        timestamp=TIMESTAMP,
    )

    assert first == second
