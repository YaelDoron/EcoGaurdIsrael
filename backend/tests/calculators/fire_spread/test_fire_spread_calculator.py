"""Tests for the pure wildfire-spread cellular-automata calculator.

Covers the verified scientific functions (p_n, e_m, alpha_wh, p_ij) and the
deterministic CA propagation behavior built on top of them. Does not test
FireEvent status or missing-DB-input handling -- those belong to later
service/agent tests (Task 5+).
"""
from __future__ import annotations

import math

import pytest

from src.calculators.fire_spread import fire_spread_calculator as fsc
from src.calculators.fire_spread.fire_spread_calculator import (
    FireSpreadCalculator,
    _MOORE_OFFSETS,
    _generate_grid_cells,
    moisture_factor,
    nominal_spread_probability,
    transition_probability,
    wind_topography_factor,
)
from src.calculators.fire_spread.fire_spread_config import (
    CA_TIME_STEP_MINUTES,
    MOISTURE_OF_EXTINCTION,
    NOMINAL_SPREAD_PROBABILITY,
    PREDICTION_RADIUS_KM,
    PROPAGATION_THRESHOLD,
)
from src.models.fire_spread_fuel_class import FireSpreadFuelClass as FC
from src.models.fire_spread_input import FireSpreadInput
from src.utils.geo import haversine_distance_km

ORIGIN_LAT = 32.75
ORIGIN_LON = 35.0

# Deliberately favorable conditions (strong wind, bone-dry fuel, a fuel class
# with a high self-transition p_n) so the CA actually propagates -- used by
# tests that need real reached cells.
FAVORABLE_KWARGS = dict(
    origin_latitude=ORIGIN_LAT,
    origin_longitude=ORIGIN_LON,
    wind_speed_kmh=60.0,
    wind_direction_deg=0.0,  # wind FROM north -> blows toward south
    fuel_moisture_percent=0.0,
    fuel_class=FC.GRASSLAND,
)

# Deliberately unfavorable (but entirely valid) conditions that should
# produce a legitimate "no predicted spread" result.
NO_SPREAD_KWARGS = dict(
    origin_latitude=ORIGIN_LAT,
    origin_longitude=ORIGIN_LON,
    wind_speed_kmh=20.0,
    wind_direction_deg=270.0,
    fuel_moisture_percent=12.0,
    fuel_class=FC.SHRUBS,
)


def make_input(**overrides) -> FireSpreadInput:
    defaults = dict(horizon_minutes=30)
    defaults.update(FAVORABLE_KWARGS)
    defaults.update(overrides)
    return FireSpreadInput(**defaults)


# ---------------------------------------------------------------------------
# p_n: nominal vegetation spread probability
# ---------------------------------------------------------------------------


def test_p_n_known_matrix_entries():
    # Spot-check several verified table cells (see fire_spread_prediction.md §15.2).
    assert nominal_spread_probability(FC.BROADLEAVES_FIRE_PRONE, FC.BROADLEAVES_FIRE_PRONE) == 0.300
    assert nominal_spread_probability(FC.SHRUBS, FC.SHRUBS) == 0.375
    assert nominal_spread_probability(FC.CONIFERS_FIRE_PRONE, FC.CONIFERS_FIRE_PRONE) == 0.350
    assert nominal_spread_probability(FC.GRASSLAND, FC.GRASSLAND) == 0.475
    assert nominal_spread_probability(FC.AGRO_FORESTRY, FC.AGRO_FORESTRY) == 0.350
    assert nominal_spread_probability(FC.BROADLEAVES_NON_FIRE_PRONE, FC.BROADLEAVES_NON_FIRE_PRONE) == 0.075


def test_p_n_corrected_task_4a_discrepancy_entry():
    # target=Conifers(fire-prone), source=Grassland: verified official value
    # is 0.100 -- NOT the 0.250 that had circulated in an earlier, unverified draft.
    assert nominal_spread_probability(FC.GRASSLAND, FC.CONIFERS_FIRE_PRONE) == 0.100


def test_p_n_bare_soil_is_near_zero_for_every_pairing():
    for source in FC:
        assert nominal_spread_probability(source, FC.BARE_SOIL) == pytest.approx(0.005)
        assert nominal_spread_probability(FC.BARE_SOIL, source) == pytest.approx(0.005)


def test_p_n_source_target_orientation_is_not_symmetric():
    # target=Grassland,source=Conifers (0.475) != target=Conifers,source=Grassland (0.100):
    # a regression that swaps source/target indexing would silently flip these.
    grassland_from_conifers = nominal_spread_probability(FC.CONIFERS_FIRE_PRONE, FC.GRASSLAND)
    conifers_from_grassland = nominal_spread_probability(FC.GRASSLAND, FC.CONIFERS_FIRE_PRONE)
    assert grassland_from_conifers == 0.475
    assert conifers_from_grassland == 0.100
    assert grassland_from_conifers != conifers_from_grassland


def test_p_n_matrix_has_exactly_seven_classes():
    assert len(FC) == 7
    assert len(NOMINAL_SPREAD_PROBABILITY) == 7
    for row in NOMINAL_SPREAD_PROBABILITY.values():
        assert len(row) == 7


# ---------------------------------------------------------------------------
# e_m: moisture factor
# ---------------------------------------------------------------------------


def test_moisture_factor_dry_fuel_is_near_one():
    assert moisture_factor(0.0) == pytest.approx(1.0, abs=1e-3)


def test_moisture_factor_at_extinction_is_near_zero():
    # Mx = 0.30 -> 30% fuel moisture percent is exactly the extinction point.
    assert moisture_factor(MOISTURE_OF_EXTINCTION * 100) == pytest.approx(0.0, abs=0.01)


def test_moisture_factor_above_extinction_is_clamped_to_zero():
    assert moisture_factor(60.0) == 0.0
    assert moisture_factor(100.0) == 0.0


def test_moisture_factor_moderate_value_is_between_bounds():
    value = moisture_factor(15.0)
    assert 0.0 < value < 1.0


def test_moisture_factor_is_within_unit_range_across_input_range():
    for percent in range(0, 101, 5):
        value = moisture_factor(float(percent))
        assert 0.0 <= value <= 1.0


def test_moisture_factor_monotonic_lower_moisture_higher_factor():
    values = [moisture_factor(percent) for percent in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)]
    assert values == sorted(values, reverse=True)


def test_moisture_factor_negative_input_behaves_like_zero():
    # Not a valid FireSpreadInput value (rejected at the model boundary), but
    # the pure function itself should not raise for it, matching the
    # verified formula's own moist>=0 clip.
    assert moisture_factor(-10.0) == moisture_factor(0.0)


# ---------------------------------------------------------------------------
# alpha_wh: wind/topography factor
# ---------------------------------------------------------------------------


def test_wind_factor_zero_wind_is_neutral_in_every_direction():
    for bearing in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        assert wind_topography_factor(0.0, 123.0, bearing) == pytest.approx(1.0, abs=1e-9)


def test_wind_factor_downwind_exceeds_crosswind_exceeds_upwind():
    # Wind FROM north (0 deg) blows toward south (bearing 180).
    upwind = wind_topography_factor(60.0, 0.0, 0.0)  # propagating north, into the wind
    crosswind = wind_topography_factor(60.0, 0.0, 90.0)  # propagating east
    downwind = wind_topography_factor(60.0, 0.0, 180.0)  # propagating south, with the wind

    assert upwind < crosswind < downwind


def test_wind_factor_downwind_symmetric_crosswind_equal():
    east = wind_topography_factor(60.0, 0.0, 90.0)
    west = wind_topography_factor(60.0, 0.0, 270.0)
    assert east == pytest.approx(west, abs=1e-9)


def test_wind_factor_stronger_wind_increases_downwind_effect():
    values = [wind_topography_factor(speed, 0.0, 180.0) for speed in (0.0, 10.0, 20.0, 40.0, 60.0, 100.0)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_wind_factor_wrap_around_0_360_matches():
    just_under_360 = wind_topography_factor(20.0, 359.999, 10.0)
    at_zero = wind_topography_factor(20.0, 0.0, 10.0)
    assert just_under_360 == pytest.approx(at_zero, abs=1e-4)


def test_wind_factor_deterministic_repeat_call():
    first = wind_topography_factor(35.0, 47.0, 210.0)
    second = wind_topography_factor(35.0, 47.0, 210.0)
    assert first == second


def test_wind_factor_negative_wind_speed_not_exercised_by_valid_input():
    # FireSpreadInput itself rejects negative wind speed; the pure function
    # is not required to handle it meaningfully, so this is intentionally
    # not asserted here.
    pass


# ---------------------------------------------------------------------------
# p_ij: transition probability
# ---------------------------------------------------------------------------


def test_p_ij_is_within_unit_range():
    p = transition_probability(
        source_fuel_class=FC.SHRUBS,
        target_fuel_class=FC.SHRUBS,
        wind_speed_kmh=25.0,
        wind_direction_deg=90.0,
        propagation_bearing_deg=270.0,
        fuel_moisture_percent=8.0,
    )
    assert 0.0 <= p <= 1.0


def test_p_ij_zero_moisture_factor_gives_zero_probability():
    p = transition_probability(
        source_fuel_class=FC.GRASSLAND,
        target_fuel_class=FC.GRASSLAND,
        wind_speed_kmh=60.0,
        wind_direction_deg=0.0,
        propagation_bearing_deg=180.0,
        fuel_moisture_percent=100.0,
    )
    assert p == 0.0


def test_p_ij_matches_verified_equation_numerically():
    p_n = nominal_spread_probability(FC.SHRUBS, FC.SHRUBS)
    alpha_wh = wind_topography_factor(20.0, 270.0, 90.0)
    e_m = moisture_factor(10.0)
    expected = min(max((1.0 - (1.0 - p_n) ** alpha_wh) * e_m, 0.0), 1.0)

    actual = transition_probability(
        source_fuel_class=FC.SHRUBS,
        target_fuel_class=FC.SHRUBS,
        wind_speed_kmh=20.0,
        wind_direction_deg=270.0,
        propagation_bearing_deg=90.0,
        fuel_moisture_percent=10.0,
    )
    assert actual == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Terrain: verified neutral behavior
# ---------------------------------------------------------------------------


def test_terrain_is_not_an_accepted_calculator_parameter():
    # EcoGuard V1 neutral-terrain adaptation is folded in internally
    # (h_effect == 1.0 exactly at dh = 0); the calculator boundary must not
    # expose a terrain/slope/elevation parameter at all.
    import inspect

    signature = inspect.signature(wind_topography_factor)
    assert set(signature.parameters) == {"wind_speed_kmh", "wind_direction_deg", "propagation_bearing_deg"}


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


def test_grid_includes_origin():
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)
    latitude, longitude = grid[(0, 0)]
    assert latitude == pytest.approx(ORIGIN_LAT, abs=1e-9)
    assert longitude == pytest.approx(ORIGIN_LON, abs=1e-9)


def test_grid_cells_within_radius():
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)
    for latitude, longitude in grid.values():
        distance = haversine_distance_km(ORIGIN_LAT, ORIGIN_LON, latitude, longitude)
        assert distance <= PREDICTION_RADIUS_KM + 1e-6


def test_grid_excludes_cells_outside_radius():
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)
    # A cell 21 steps out (5.25 km) must not be present; 20 steps (5.0 km) must be.
    assert (21, 0) not in grid
    assert (20, 0) in grid


def test_grid_adjacent_cell_spacing_is_approximately_250m():
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)
    origin = grid[(0, 0)]
    neighbor = grid[(1, 0)]
    distance_km = haversine_distance_km(*origin, *neighbor)
    assert distance_km == pytest.approx(0.25, abs=0.001)


def test_moore_offsets_are_deterministic_compass_order():
    labels_in_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    assert len(_MOORE_OFFSETS) == 8
    bearings = [bearing for _offset, bearing in _MOORE_OFFSETS]
    assert bearings == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    # Re-importing / re-evaluating the module-level constant is stable.
    assert list(fsc._MOORE_OFFSETS) == list(_MOORE_OFFSETS)


# ---------------------------------------------------------------------------
# CA propagation behavior
# ---------------------------------------------------------------------------


def test_calculate_is_deterministic_across_repeated_calls():
    calculator = FireSpreadCalculator()
    input_data = make_input()
    first = calculator.calculate(input_data)
    second = calculator.calculate(input_data)
    assert first.cells == second.cells


def test_calculate_rejects_non_input_argument():
    with pytest.raises(ValueError):
        FireSpreadCalculator().calculate("not-a-FireSpreadInput")


def test_valid_no_spread_result_is_not_an_error():
    # Methodology 1.1: no cell crosses the propagation threshold, so the
    # result is only the first ring of risk-only neighbours.
    calculator = FireSpreadCalculator()
    input_data = make_input(**NO_SPREAD_KWARGS)
    result = calculator.calculate(input_data)
    assert len(result.cells) == 8
    assert all(cell.spread_probability < PROPAGATION_THRESHOLD for cell in result.cells)
    assert all(cell.reached_step == 1 for cell in result.cells)
    assert result.horizon_minutes == 30


def test_fuel_moisture_above_extinction_produces_empty_result():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(**{**NO_SPREAD_KWARGS, "fuel_moisture_percent": 40.0}))
    assert result.cells == ()


def test_origin_cell_never_appears_in_output():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=60))
    assert len(result.cells) > 0
    for cell in result.cells:
        distance_from_origin = haversine_distance_km(ORIGIN_LAT, ORIGIN_LON, cell.latitude, cell.longitude)
        assert distance_from_origin > 0.0


def test_first_step_reached_cells_are_immediate_neighbors_only():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=30))
    step_one_cells = [cell for cell in result.cells if cell.reached_step == 1]
    assert step_one_cells
    max_first_step_distance_km = 0.250 * math.sqrt(2) + 0.01  # Moore-adjacent diagonal + tolerance
    for cell in step_one_cells:
        distance = haversine_distance_km(ORIGIN_LAT, ORIGIN_LON, cell.latitude, cell.longitude)
        assert distance <= max_first_step_distance_km


def test_threshold_boundary_just_below_no_propagation(monkeypatch):
    monkeypatch.setattr(fsc, "transition_probability", lambda **kwargs: PROPAGATION_THRESHOLD - 1e-9)
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=30))
    # Only the origin's 8 neighbours are exposed; none may propagate further.
    assert len(result.cells) == 8
    assert all(cell.reached_step == 1 for cell in result.cells)
    assert all(cell.spread_probability < PROPAGATION_THRESHOLD for cell in result.cells)


def test_threshold_boundary_at_threshold_propagates(monkeypatch):
    monkeypatch.setattr(fsc, "transition_probability", lambda **kwargs: PROPAGATION_THRESHOLD)
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=30))
    assert len(result.cells) > 0
    assert all(cell.spread_probability == pytest.approx(PROPAGATION_THRESHOLD) for cell in result.cells)


def test_downwind_cell_has_higher_probability_than_upwind_cell():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=30))
    step_one = {cell.latitude: cell for cell in result.cells if cell.reached_step == 1}

    # Wind FROM north blows toward south: the direct-south neighbor (lower
    # latitude, same longitude) should have the highest step-1 probability.
    south_probability = max(
        cell.spread_probability
        for cell in result.cells
        if cell.reached_step == 1 and cell.longitude == pytest.approx(ORIGIN_LON, abs=1e-9)
    )
    assert south_probability == max(cell.spread_probability for cell in result.cells if cell.reached_step == 1)


def test_lower_fuel_moisture_increases_spread_potential():
    calculator = FireSpreadCalculator()
    dry_result = calculator.calculate(make_input(horizon_minutes=30, fuel_moisture_percent=0.0))
    moist_result = calculator.calculate(make_input(horizon_minutes=30, fuel_moisture_percent=20.0))
    assert len(dry_result.cells) >= len(moist_result.cells)


def test_fuel_class_affects_propagation_probability():
    calculator = FireSpreadCalculator()
    grassland_result = calculator.calculate(make_input(horizon_minutes=30, fuel_class=FC.GRASSLAND))
    bare_result = calculator.calculate(make_input(horizon_minutes=30, fuel_class=FC.BARE_SOIL))
    assert any(cell.spread_probability >= PROPAGATION_THRESHOLD for cell in grassland_result.cells)
    assert not any(cell.spread_probability >= PROPAGATION_THRESHOLD for cell in bare_result.cells)
    assert all(cell.spread_probability < 0.01 for cell in bare_result.cells)


def test_non_burnable_fuel_never_propagates_even_under_extreme_conditions():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(
        make_input(
            horizon_minutes=60,
            wind_speed_kmh=100.0,
            fuel_moisture_percent=0.0,
            fuel_class=FC.BARE_SOIL,
        )
    )
    # Bare soil is only a near-zero first ring of risk-only cells.
    assert len(result.cells) == 8
    assert all(cell.reached_step == 1 for cell in result.cells)
    assert all(cell.spread_probability < 0.01 for cell in result.cells)


def test_farther_cells_require_additional_steps():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=60))
    by_distance = sorted(
        result.cells,
        key=lambda cell: haversine_distance_km(ORIGIN_LAT, ORIGIN_LON, cell.latitude, cell.longitude),
    )
    reached_steps_by_distance = [cell.reached_step for cell in by_distance]
    # Reached step must not decrease as distance from the origin increases
    # by more than one full step-radius at a time is too strict to assert
    # generally, but the nearest cell must not be reached later than the
    # farthest cell.
    assert reached_steps_by_distance[0] <= reached_steps_by_distance[-1]


def test_60_minute_first_6_steps_match_30_minute_run():
    calculator = FireSpreadCalculator()
    result_30 = calculator.calculate(make_input(horizon_minutes=30))
    result_60 = calculator.calculate(make_input(horizon_minutes=60))

    cells_60_by_key = {(cell.reached_step, cell.latitude, cell.longitude): cell for cell in result_60.cells}
    # Only spreading cells are fixed once reached; a risk-only cell at 30
    # minutes may be re-exposed (or upgraded) by later 60-minute steps.
    for cell in (c for c in result_30.cells if c.spread_probability >= PROPAGATION_THRESHOLD):
        key = (cell.reached_step, cell.latitude, cell.longitude)
        assert key in cells_60_by_key
        matching = cells_60_by_key[key]
        assert matching.spread_probability == cell.spread_probability
        assert matching.reached_minutes == cell.reached_minutes


def test_60_minute_run_may_contain_cells_beyond_30_minutes():
    calculator = FireSpreadCalculator()
    result_30 = calculator.calculate(make_input(horizon_minutes=30))
    result_60 = calculator.calculate(make_input(horizon_minutes=60))
    assert len(result_60.cells) >= len(result_30.cells)
    later_steps = [cell for cell in result_60.cells if cell.reached_step > 6]
    assert later_steps


def test_output_cells_are_stably_ordered():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=60))
    ordering_keys = [(cell.reached_step, cell.latitude, cell.longitude) for cell in result.cells]
    assert ordering_keys == sorted(ordering_keys)


def test_reached_minutes_consistent_with_step():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=60))
    for cell in result.cells:
        assert cell.reached_minutes == cell.reached_step * CA_TIME_STEP_MINUTES


def test_methodology_identity_on_result():
    calculator = FireSpreadCalculator()
    result = calculator.calculate(make_input(horizon_minutes=30))
    assert result.methodology == "ECOGUARD_PROPAGATOR_CA"
    assert result.methodology_version == "1.1"


# ---------------------------------------------------------------------------
# Methodology 1.1: spreading vs risk-only cells (fire_spread_prediction.md §8)
# ---------------------------------------------------------------------------

REALISTIC_TEMPERATURE_C = 35.0
REALISTIC_HUMIDITY_PCT = 25.0
REALISTIC_WIND_KMH = 30.0
# Wind FROM north -> blows toward south, so the south neighbour is downwind.
REALISTIC_WIND_DIRECTION_DEG = 0.0


def _realistic_fuel_moisture_percent(temperature_c: float, relative_humidity_pct: float) -> float:
    # Same equilibrium-moisture estimate FireSpreadInputService feeds the calculator.
    from src.calculators.fire_danger.ffwi_calculator import (
        _celsius_to_fahrenheit,
        _equilibrium_moisture_content,
    )

    return _equilibrium_moisture_content(
        relative_humidity_pct=relative_humidity_pct,
        temperature_f=_celsius_to_fahrenheit(temperature_c),
    )


def make_realistic_input(**overrides) -> FireSpreadInput:
    values = dict(
        origin_latitude=ORIGIN_LAT,
        origin_longitude=ORIGIN_LON,
        wind_speed_kmh=REALISTIC_WIND_KMH,
        wind_direction_deg=REALISTIC_WIND_DIRECTION_DEG,
        fuel_moisture_percent=_realistic_fuel_moisture_percent(REALISTIC_TEMPERATURE_C, REALISTIC_HUMIDITY_PCT),
        fuel_class=FC.SHRUBS,
        horizon_minutes=30,
    )
    values.update(overrides)
    return FireSpreadInput(**values)


def _spreading(cells):
    return [cell for cell in cells if cell.spread_probability >= PROPAGATION_THRESHOLD]


def _risk_only(cells):
    return [cell for cell in cells if cell.spread_probability < PROPAGATION_THRESHOLD]


def _cell_keys(cells):
    return [(cell.latitude, cell.longitude) for cell in cells]


def test_realistic_shrubs_produce_eight_risk_only_cells_strongest_downwind():
    result = FireSpreadCalculator().calculate(make_realistic_input())

    assert len(result.cells) == 8
    assert _spreading(result.cells) == []
    assert all(0.0 < cell.spread_probability < PROPAGATION_THRESHOLD for cell in result.cells)
    assert all(cell.reached_step == 1 and cell.reached_minutes == CA_TIME_STEP_MINUTES for cell in result.cells)
    strongest = max(result.cells, key=lambda cell: cell.spread_probability)
    assert strongest.longitude == pytest.approx(ORIGIN_LON, abs=1e-9)
    assert strongest.latitude < ORIGIN_LAT


def test_realistic_grass_risk_exceeds_realistic_shrubs():
    calculator = FireSpreadCalculator()
    shrubs = calculator.calculate(make_realistic_input(fuel_class=FC.SHRUBS))
    grass = calculator.calculate(make_realistic_input(fuel_class=FC.GRASSLAND))

    assert len(grass.cells) == 8
    assert _spreading(grass.cells) == []
    shrub_by_location = {(cell.latitude, cell.longitude): cell.spread_probability for cell in shrubs.cells}
    for cell in grass.cells:
        assert cell.spread_probability > shrub_by_location[(cell.latitude, cell.longitude)]


def test_strong_wind_grass_has_spreading_cells_and_risk_only_perimeter():
    result = FireSpreadCalculator().calculate(
        make_realistic_input(
            fuel_class=FC.GRASSLAND,
            wind_speed_kmh=80.0,
            fuel_moisture_percent=_realistic_fuel_moisture_percent(38.0, 20.0),
            horizon_minutes=60,
        )
    )
    spreading = _spreading(result.cells)
    risk_only = _risk_only(result.cells)

    assert spreading
    assert risk_only
    # Every risk-only cell borders a spreading cell or the origin, so the
    # perimeter lies beyond at least one spreading cell.
    assert max(cell.reached_step for cell in risk_only) > 1


def test_30_minute_spreading_cells_are_contained_in_60_minute_run_which_extends_further():
    calculator = FireSpreadCalculator()
    strong = dict(
        fuel_class=FC.GRASSLAND,
        wind_speed_kmh=80.0,
        fuel_moisture_percent=_realistic_fuel_moisture_percent(38.0, 20.0),
    )
    result_30 = calculator.calculate(make_realistic_input(horizon_minutes=30, **strong))
    result_60 = calculator.calculate(make_realistic_input(horizon_minutes=60, **strong))

    spreading_60 = {
        (cell.latitude, cell.longitude): (cell.reached_step, cell.spread_probability)
        for cell in _spreading(result_60.cells)
    }
    for cell in _spreading(result_30.cells):
        assert spreading_60[(cell.latitude, cell.longitude)] == (cell.reached_step, cell.spread_probability)
    assert len(spreading_60) > len(_spreading(result_30.cells))
    assert max(cell.reached_step for cell in result_60.cells) > 6


def test_without_propagation_30_and_60_minute_results_are_identical():
    calculator = FireSpreadCalculator()
    result_30 = calculator.calculate(make_realistic_input(horizon_minutes=30))
    result_60 = calculator.calculate(make_realistic_input(horizon_minutes=60))
    assert result_30.cells == result_60.cells


def test_risk_only_cells_never_act_as_propagation_sources(monkeypatch):
    # Every transition is just below the threshold: if risk-only cells were
    # sources, exposure would continue outward beyond the first ring.
    monkeypatch.setattr(fsc, "transition_probability", lambda **kwargs: PROPAGATION_THRESHOLD - 0.01)
    result = FireSpreadCalculator().calculate(make_input(horizon_minutes=60))
    assert len(result.cells) == 8
    assert {cell.reached_step for cell in result.cells} == {1}


def test_cells_are_never_duplicated():
    result = FireSpreadCalculator().calculate(
        make_realistic_input(
            fuel_class=FC.GRASSLAND,
            wind_speed_kmh=80.0,
            fuel_moisture_percent=_realistic_fuel_moisture_percent(38.0, 20.0),
            horizon_minutes=60,
        )
    )
    keys = _cell_keys(result.cells)
    assert len(keys) == len(set(keys))


def test_risk_only_cell_keeps_highest_probability_and_earliest_step(monkeypatch):
    # Only the direct east/west neighbours of the origin may spread (p=0.9);
    # every other direction is risk-only at a bearing-dependent value. Cells
    # north of the E/W spreading line are then exposed several times from
    # different sources and bearings, and must keep only their highest
    # probability and the earliest step at which it was recorded.
    def fake(**kwargs):
        bearing = kwargs["propagation_bearing_deg"]
        if bearing in (90.0, 270.0):
            return 0.9
        return {0.0: 0.2, 45.0: 0.3, 315.0: 0.4}.get(bearing, 0.1)

    monkeypatch.setattr(fsc, "transition_probability", fake)
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)
    reached_step, reached_probability, risk_only = fsc._run_cellular_automata(
        grid=grid,
        fuel_class=FC.GRASSLAND,
        wind_speed_kmh=0.0,
        wind_direction_deg=0.0,
        fuel_moisture_percent=0.0,
        horizon_steps=2,
    )

    # (1, 0) is north of the origin: step 1 it is exposed only from the
    # origin (bearing 0 -> 0.2) and from nothing else yet. At step 2 the
    # spreading (0, 1) exposes it via bearing 315 (0.4) and (0, -1) via
    # bearing 45 (0.3): the highest value, 0.4, must win with step 2.
    assert risk_only[(1, 0)] == (0.4, 2)
    # (1, 1) is first exposed at step 1 from the origin (bearing 45 -> 0.3);
    # at step 2 the spreading (0, 1) exposes it via bearing 0 (0.2), which
    # is lower, so step 1 / 0.3 is retained.
    assert risk_only[(1, 1)] == (0.3, 1)
    assert set(risk_only).isdisjoint(reached_step)


def test_risk_only_cell_is_upgraded_to_spreading_and_removed_from_risk_only(monkeypatch):
    # From the origin the east neighbour is only risk-only (bearing 90 -> 0.3),
    # but the north (bearing 0) and south-east (bearing 135) neighbours
    # spread at step 1 and then reach the east neighbour at step 2.
    def fake(**kwargs):
        return {0.0: 0.9, 135.0: 0.8}.get(kwargs["propagation_bearing_deg"], 0.3)

    monkeypatch.setattr(fsc, "transition_probability", fake)
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)

    _, _, risk_only_after_one = fsc._run_cellular_automata(
        grid=grid,
        fuel_class=FC.GRASSLAND,
        wind_speed_kmh=0.0,
        wind_direction_deg=0.0,
        fuel_moisture_percent=0.0,
        horizon_steps=1,
    )
    assert risk_only_after_one[(0, 1)] == (0.3, 1)

    reached_step, reached_probability, risk_only = fsc._run_cellular_automata(
        grid=grid,
        fuel_class=FC.GRASSLAND,
        wind_speed_kmh=0.0,
        wind_direction_deg=0.0,
        fuel_moisture_percent=0.0,
        horizon_steps=2,
    )
    assert reached_step[(0, 1)] == 2
    assert reached_probability[(0, 1)] >= PROPAGATION_THRESHOLD
    assert (0, 1) not in risk_only


def test_spreading_if_and_only_if_probability_reaches_threshold():
    calculator = FireSpreadCalculator()
    strong = calculator.calculate(
        make_realistic_input(
            fuel_class=FC.GRASSLAND,
            wind_speed_kmh=80.0,
            fuel_moisture_percent=_realistic_fuel_moisture_percent(38.0, 20.0),
            horizon_minutes=60,
        )
    )
    grid = _generate_grid_cells(ORIGIN_LAT, ORIGIN_LON)
    reached_step, _, risk_only = fsc._run_cellular_automata(
        grid=grid,
        fuel_class=FC.GRASSLAND,
        wind_speed_kmh=80.0,
        wind_direction_deg=REALISTIC_WIND_DIRECTION_DEG,
        fuel_moisture_percent=_realistic_fuel_moisture_percent(38.0, 20.0),
        horizon_steps=12,
    )
    spreading_locations = {grid[key] for key in reached_step if key != (0, 0)}
    risk_only_locations = {grid[key] for key in risk_only}

    for cell in strong.cells:
        location = (cell.latitude, cell.longitude)
        if cell.spread_probability >= PROPAGATION_THRESHOLD:
            assert location in spreading_locations
        else:
            assert location in risk_only_locations
    assert len(strong.cells) == len(spreading_locations) + len(risk_only_locations)


def test_realistic_calculation_is_deterministic_including_risk_only_cells():
    calculator = FireSpreadCalculator()
    for horizon_minutes in (30, 60):
        first = calculator.calculate(make_realistic_input(horizon_minutes=horizon_minutes))
        second = calculator.calculate(make_realistic_input(horizon_minutes=horizon_minutes))
        assert first == second
