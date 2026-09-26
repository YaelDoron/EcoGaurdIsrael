"""Calibration/audit tests for the spread PROPAGATION_THRESHOLD (analysis only).

Candidate thresholds are evaluated through scripts.calibrate_fire_spread_threshold,
which patches the calculator's constant only inside a context manager. The
production threshold itself is asserted UNCHANGED here - recalibration is a
separate, approved change.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.calibrate_fire_spread_threshold import (
    BURNABLE_FUELS,
    CANDIDATE_THRESHOLDS,
    DIRECTIONS,
    REGIMES,
    WeatherPoint,
    directional_probability,
    fuel_moisture_percent,
    minimum_downwind_wind_kmh,
    regime_weather_grid,
    reproduce_p_ij,
    run_ca,
    severe_corner,
    threshold_metrics,
)
from src.agents.response import chatbot_agent
from src.calculators.fire_spread import fire_spread_calculator as fsc
from src.calculators.fire_spread.fire_spread_calculator import (
    moisture_factor,
    nominal_spread_probability,
    wind_topography_factor,
)
from src.calculators.fire_spread.fire_spread_config import PROPAGATION_THRESHOLD
from src.models.fire_spread_fuel_class import FireSpreadFuelClass as FC
from src.services.fire_spread.fire_spread_input_service import _equilibrium_moisture_percent

ALL_WEATHER = tuple(w for regime, _ in REGIMES for w in regime_weather_grid(regime))
# The pre-calibration value, kept as explicit evidence of why it was rejected (Task 11C).
PREVIOUS_THRESHOLD = 0.50
EXTREME_REALISTIC = WeatherPoint("extreme", temperature_c=41.0, relative_humidity=10.0, wind_speed_kmh=60.0)


def test_production_threshold_is_the_calibrated_candidate_everywhere():
    assert PROPAGATION_THRESHOLD == 0.45
    assert fsc.PROPAGATION_THRESHOLD == 0.45
    assert chatbot_agent._SPREAD_PROPAGATION_THRESHOLD is PROPAGATION_THRESHOLD


def test_candidate_evaluation_never_leaks_into_production_constant():
    run_ca(FC.GRASSLAND, severe_corner("active_fire"), 30, 0.30)
    assert fsc.PROPAGATION_THRESHOLD == 0.45


# --- formula reproduction ---------------------------------------------------


@pytest.mark.parametrize("fuel", list(FC))
def test_p_ij_formula_is_reproduced_exactly(fuel):
    for weather in ALL_WEATHER:
        for _, relative in DIRECTIONS:
            bearing = (180.0 + relative) % 360.0
            expected = reproduce_p_ij(
                nominal_spread_probability(fuel, fuel),
                wind_topography_factor(weather.wind_speed_kmh, 0.0, bearing),
                moisture_factor(weather.fuel_moisture_percent),
            )
            assert directional_probability(fuel, weather, relative) == pytest.approx(expected, abs=1e-12)


def test_calibration_fuel_moisture_matches_the_spread_input_service():
    for weather in ALL_WEATHER:
        observation = SimpleNamespace(temperature=weather.temperature_c, relative_humidity=weather.relative_humidity)
        assert fuel_moisture_percent(weather.temperature_c, weather.relative_humidity) == pytest.approx(
            _equilibrium_moisture_percent(observation)
        )


# --- deterministic matrix / current-threshold feasibility -------------------


def test_calibration_matrix_is_deterministic():
    assert [threshold_metrics(t) for t in CANDIDATE_THRESHOLDS] == [threshold_metrics(t) for t in CANDIDATE_THRESHOLDS]


def test_previous_threshold_only_ever_propagated_grassland_in_the_simulator_range():
    propagating = {
        fuel
        for fuel in FC
        for weather in ALL_WEATHER
        for _, relative in DIRECTIONS
        if directional_probability(fuel, weather, relative) >= PREVIOUS_THRESHOLD
    }
    assert propagating == {FC.GRASSLAND}


@pytest.mark.parametrize(
    "fuel, expected_kmh",
    [(FC.SHRUBS, 54), (FC.GRASSLAND, 17), (FC.CONIFERS_FIRE_PRONE, None), (FC.AGRO_FORESTRY, None)],
)
def test_minimum_downwind_wind_at_previous_threshold_and_observed_jerusalem_moisture(fuel, expected_kmh):
    assert minimum_downwind_wind_kmh(fuel, 2.5306, PREVIOUS_THRESHOLD) == expected_kmh


# --- directional ordering and fuel-class differences ------------------------


@pytest.mark.parametrize("fuel", BURNABLE_FUELS)
def test_downwind_beats_crosswind_beats_upwind_whenever_there_is_wind(fuel):
    for weather in ALL_WEATHER:
        if weather.wind_speed_kmh == 0:
            continue
        down, cross, up = (directional_probability(fuel, weather, d) for _, d in DIRECTIONS)
        assert down > cross > up


@pytest.mark.parametrize("fuel", BURNABLE_FUELS)
def test_direction_is_material_not_just_ordered_at_simulator_winds(fuel):
    for weather in ALL_WEATHER:
        if weather.wind_speed_kmh >= 20:
            assert directional_probability(fuel, weather, 0.0) - directional_probability(fuel, weather, 180.0) > 0.09


def test_fuel_class_ordering_is_stable_across_all_weather():
    for weather in ALL_WEATHER:
        p = {fuel: directional_probability(fuel, weather, 0.0) for fuel in FC}
        assert p[FC.GRASSLAND] > p[FC.SHRUBS] > p[FC.CONIFERS_FIRE_PRONE] > p[FC.BROADLEAVES_FIRE_PRONE]
        assert p[FC.CONIFERS_FIRE_PRONE] == pytest.approx(p[FC.AGRO_FORESTRY])
        assert p[FC.BROADLEAVES_FIRE_PRONE] > p[FC.BROADLEAVES_NON_FIRE_PRONE] > p[FC.BARE_SOIL]


# --- recommended candidate (0.45, Task 12 re-evaluation) behavioural expectations

RECOMMENDED = 0.45


def test_recommended_candidate_is_the_production_threshold():
    assert RECOMMENDED == PROPAGATION_THRESHOLD


def test_low_risk_never_and_moderate_rarely_propagate_at_recommended_candidate():
    metrics = threshold_metrics(RECOMMENDED)
    assert metrics.low_false_propagation == 0.0
    assert metrics.moderate_downwind_rate == pytest.approx(0.0444, abs=1e-3)  # grass only, limited
    assert metrics.strong_upwind_rate == 0.0
    assert metrics.directional_consistency == 1.0
    assert set(metrics.fuels_propagating_under_strong) == {"GRASSLAND", "SHRUBS", "AGRO_FORESTRY", "CONIFERS_FIRE_PRONE"}


def test_strong_fire_shrubs_propagate_at_recommended_candidate_but_not_at_previous():
    weather = severe_corner("active_fire")
    assert run_ca(FC.SHRUBS, weather, 30, RECOMMENDED).propagated_cells > 0
    assert run_ca(FC.SHRUBS, weather, 30, PREVIOUS_THRESHOLD).propagated_cells == 0


def test_strong_simulator_wind_gives_directional_shrub_propagation_that_grows_from_30_to_60():
    at_30 = run_ca(FC.SHRUBS, severe_corner("active_fire"), 30, RECOMMENDED)
    at_60 = run_ca(FC.SHRUBS, severe_corner("active_fire"), 60, RECOMMENDED)

    assert at_30.propagated_cells > 0
    assert at_30.upwind_cells == 0 and at_60.upwind_cells == 0
    assert at_60.propagated_cells > at_30.propagated_cells
    assert at_60.max_distance_m > at_30.max_distance_m
    assert at_60.max_reached_step > at_30.max_reached_step


def test_observed_jerusalem_forest_shrubs_propagate_only_downwind_at_recommended_candidate():
    observed = WeatherPoint("observed", temperature_c=36.6, relative_humidity=10.9, wind_speed_kmh=27.5)
    at_30 = run_ca(FC.SHRUBS, observed, 30, RECOMMENDED)
    at_60 = run_ca(FC.SHRUBS, observed, 60, RECOMMENDED)
    assert (at_30.propagated_cells, at_60.propagated_cells) == (6, 12)
    assert at_60.upwind_cells == 0
    assert run_ca(FC.SHRUBS, observed, 60, PREVIOUS_THRESHOLD).propagated_cells == 0


def test_bare_soil_never_propagates_at_any_candidate():
    for threshold in CANDIDATE_THRESHOLDS:
        assert run_ca(FC.BARE_SOIL, EXTREME_REALISTIC, 60, threshold).propagated_cells == 0
