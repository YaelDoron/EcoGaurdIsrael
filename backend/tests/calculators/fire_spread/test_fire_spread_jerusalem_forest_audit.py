"""Regression pins for the Jerusalem Forest spread audit (scripts/audit_fire_spread.py).

The inputs below are the exact persisted values the demo pipeline fed the
calculator for the observed CONFIRMED Jerusalem Forest event (severity
assessment vegetation snapshot 'Shrub cover' from Copernicus, and the traced
simulated weather observation). They document WHY that event legitimately
stayed "spread risk only", and prove the same calculator does propagate when
the methodology's conditions are met - without touching any threshold.
"""
from __future__ import annotations

import pytest

from src.calculators.fire_spread.fire_spread_calculator import (
    FireSpreadCalculator,
    moisture_factor,
    transition_probability,
)
from src.calculators.fire_spread.fire_spread_config import PROPAGATION_THRESHOLD
from src.models.fire_spread_fuel_class import FireSpreadFuelClass as FC
from src.models.fire_spread_input import FireSpreadInput
from src.services.fire_spread.fire_spread_input_service import _map_dominant_land_cover

JERUSALEM_FOREST_ORIGIN = (31.782972, 35.136200)
# Weather observation traced by the event's severity assessment: 36.6C, RH 10.9%,
# wind 27.5 km/h FROM 143.2 deg -> FFWI equilibrium moisture 2.53%.
JERUSALEM_FOREST_WIND_KMH = 27.5
JERUSALEM_FOREST_WIND_FROM_DEG = 143.2
JERUSALEM_FOREST_FUEL_MOISTURE_PERCENT = 2.5306


def _jerusalem_input(horizon_minutes: int, **overrides) -> FireSpreadInput:
    values = dict(
        origin_latitude=JERUSALEM_FOREST_ORIGIN[0],
        origin_longitude=JERUSALEM_FOREST_ORIGIN[1],
        wind_speed_kmh=JERUSALEM_FOREST_WIND_KMH,
        wind_direction_deg=JERUSALEM_FOREST_WIND_FROM_DEG,
        fuel_moisture_percent=JERUSALEM_FOREST_FUEL_MOISTURE_PERCENT,
        fuel_class=_map_dominant_land_cover("Shrub cover"),
        horizon_minutes=horizon_minutes,
    )
    values.update(overrides)
    return FireSpreadInput(**values)


def test_production_threshold_is_the_calibrated_value():
    assert PROPAGATION_THRESHOLD == 0.45


def test_copernicus_shrub_cover_reaches_the_calculator_as_shrubs():
    assert _jerusalem_input(30).fuel_class is FC.SHRUBS


@pytest.mark.parametrize("horizon, expected_propagated", [(30, 6), (60, 12)])
def test_jerusalem_forest_observed_case_propagates_a_narrow_northwest_finger(horizon, expected_propagated):
    # Real inputs (Copernicus 'Shrub cover', observed weather) at the calibrated 0.45:
    # only the downwind (NW, wind FROM 143.2 deg) diagonal propagates, one cell per step.
    result = FireSpreadCalculator().calculate(_jerusalem_input(horizon))
    spreading = [cell for cell in result.cells if cell.spread_probability >= PROPAGATION_THRESHOLD]

    assert len(spreading) == expected_propagated
    assert max(cell.reached_step for cell in spreading) == horizon // 5
    lat0, lon0 = JERUSALEM_FOREST_ORIGIN
    assert all(cell.latitude > lat0 and cell.longitude < lon0 for cell in spreading)  # all NW of origin
    best_first_ring = max(c.spread_probability for c in result.cells if c.reached_step == 1)
    assert best_first_ring == pytest.approx(0.4562, abs=5e-4)


def test_jerusalem_forest_best_cell_is_downwind():
    # Wind FROM 143.2 deg (SE) blows toward ~323 deg: the NW neighbour must score highest.
    result = FireSpreadCalculator().calculate(_jerusalem_input(30))
    best = max(result.cells, key=lambda cell: cell.spread_probability)
    assert best.latitude > JERUSALEM_FOREST_ORIGIN[0]
    assert best.longitude < JERUSALEM_FOREST_ORIGIN[1]


def test_jerusalem_forest_audit_is_deterministic():
    first = FireSpreadCalculator().calculate(_jerusalem_input(60))
    second = FireSpreadCalculator().calculate(_jerusalem_input(60))
    assert first == second


def test_observed_case_is_a_narrow_margin_case_not_a_forced_outcome():
    # Straight downwind at 27.5 km/h: observed 2.53% moisture (e_m = 0.893) gives p = 0.458,
    # just above the calibrated 0.45 (it would have been below the previous 0.50);
    # moisture still matters - wetter fuel (EMC 5%) falls back below the threshold.
    assert moisture_factor(0.0) == pytest.approx(1.0)
    observed = transition_probability(
        FC.SHRUBS, FC.SHRUBS, JERUSALEM_FOREST_WIND_KMH, 0.0, 180.0, JERUSALEM_FOREST_FUEL_MOISTURE_PERCENT
    )
    wetter = transition_probability(FC.SHRUBS, FC.SHRUBS, JERUSALEM_FOREST_WIND_KMH, 0.0, 180.0, 5.0)
    assert PROPAGATION_THRESHOLD <= observed < 0.50
    assert wetter < PROPAGATION_THRESHOLD


def test_positive_control_same_location_propagates_under_strong_dry_wind():
    # Positive control with realistic Israeli extreme fire weather (strong easterly/Sharav-type wind
    # ~65 km/h, same observed 2.5% fuel moisture) on grass cover - the fuel class with the highest
    # same-class p_n - at the SAME origin. Threshold untouched.
    result = FireSpreadCalculator().calculate(
        _jerusalem_input(60, fuel_class=FC.GRASSLAND, wind_speed_kmh=65.0)
    )
    spreading = [cell for cell in result.cells if cell.spread_probability >= PROPAGATION_THRESHOLD]
    assert spreading, "calculator must be able to propagate under supporting conditions"
    assert max(cell.reached_step for cell in spreading) >= 2  # fire front advanced past the first ring
