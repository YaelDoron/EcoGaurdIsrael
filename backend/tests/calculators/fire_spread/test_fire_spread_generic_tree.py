"""Generic tree-cover fuel policy (Task 14).

Copernicus provides only a generic Tree_Cover_Fraction - no leaf type, no
fire-proneness. EcoGuard therefore models "Tree cover" as GENERIC_TREE, an
EcoGuard-derived class (NOT a PROPAGATOR class): the equal-weight mixture of
PROPAGATOR's two fire-prone tree classes (fire-prone conifers, fire-prone
broadleaves). PROPAGATOR's non-fire-prone broadleaves class ("faggete",
beech woods) is excluded as a documented domain assumption: beech forest does
not occur in Israel's Mediterranean flora.
"""
from __future__ import annotations

import itertools

import pytest

from src.calculators.fire_spread.fire_spread_calculator import (
    FireSpreadCalculator,
    nominal_spread_probability,
    transition_probability,
)
from src.calculators.fire_spread.fire_spread_config import (
    GENERIC_TREE_SOURCE_CLASSES,
    METHODOLOGY_VERSION,
    NOMINAL_SPREAD_PROBABILITY,
    VERIFIED_PROPAGATOR_P_N,
)
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.models.fire_spread_fuel_class import VERIFIED_PROPAGATOR_FUEL_CLASSES, FireSpreadFuelClass as FC
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason as Reason
from src.models.fire_spread_prediction import PROPAGATION_THRESHOLD
from src.services.fire_spread.fire_spread_input_service import _map_dominant_land_cover
from src.simulation.generators.weather_data_generator import WEATHER_SCENARIO_PROFILES
from src.simulation.scenario_type import ScenarioType
from tests.services.fire_spread.test_fire_spread_input_service import AS_OF
from tests.services.fire_spread.test_fire_spread_input_service import build_service as build_spread_service
from tests.services.fire_spread.test_fire_spread_input_service import make_assessment

ORIGIN = (32.9150, 35.3450)  # Galilee (Har Kamon) canonical anchor - Tree-dominant after Task 13
CONIFERS, BROADLEAVES = FC.CONIFERS_FIRE_PRONE, FC.BROADLEAVES_FIRE_PRONE


def _emc(temperature_c: float, humidity: float) -> float:
    from src.calculators.fire_danger.ffwi_calculator import _celsius_to_fahrenheit, _equilibrium_moisture_content

    return _equilibrium_moisture_content(relative_humidity_pct=humidity, temperature_f=_celsius_to_fahrenheit(temperature_c))


def _run(temperature_c, humidity, wind_kmh, horizon=60):
    cells = FireSpreadCalculator().calculate(
        FireSpreadInput(
            origin_latitude=ORIGIN[0], origin_longitude=ORIGIN[1], wind_speed_kmh=wind_kmh,
            wind_direction_deg=0.0, fuel_moisture_percent=_emc(temperature_c, humidity),
            fuel_class=FC.GENERIC_TREE, horizon_minutes=horizon,
        )
    ).cells
    spreading = [c for c in cells if c.spread_probability >= PROPAGATION_THRESHOLD]
    return spreading, [c for c in cells if c.spread_probability < PROPAGATION_THRESHOLD]


def _grid(scenario):
    p = WEATHER_SCENARIO_PROFILES[scenario]
    spans = [(lo, (lo + hi) / 2, hi) for lo, hi in (p.temperature_celsius, p.relative_humidity_percent, p.wind_speed_kmh)]
    return list(itertools.product(*spans))


# --- Mapping and derivation ------------------------------------------------------


def test_copernicus_tree_cover_maps_to_the_generic_tree_fuel():
    assert _map_dominant_land_cover("Tree cover") is FC.GENERIC_TREE


def test_generic_tree_is_ecoguard_derived_not_a_propagator_class():
    assert FC.GENERIC_TREE not in VERIFIED_PROPAGATOR_FUEL_CLASSES
    assert len(VERIFIED_PROPAGATOR_FUEL_CLASSES) == 7
    assert set(GENERIC_TREE_SOURCE_CLASSES) == {CONIFERS, BROADLEAVES}


def test_verified_propagator_table_is_unchanged():
    assert set(VERIFIED_PROPAGATOR_P_N) == VERIFIED_PROPAGATOR_FUEL_CLASSES
    assert VERIFIED_PROPAGATOR_P_N[CONIFERS][CONIFERS] == 0.350
    assert VERIFIED_PROPAGATOR_P_N[BROADLEAVES][BROADLEAVES] == 0.300
    assert VERIFIED_PROPAGATOR_P_N[CONIFERS][FC.GRASSLAND] == 0.100  # [target][source]: Task 4A corrected entry
    for target in VERIFIED_PROPAGATOR_FUEL_CLASSES:
        for source in VERIFIED_PROPAGATOR_FUEL_CLASSES:
            assert NOMINAL_SPREAD_PROBABILITY[target][source] == VERIFIED_PROPAGATOR_P_N[target][source]


def test_generic_tree_self_transition_is_the_mean_of_the_fire_prone_tree_diagonals():
    assert nominal_spread_probability(FC.GENERIC_TREE, FC.GENERIC_TREE) == pytest.approx((0.350 + 0.300) / 2)
    # Bracketed by the known fire-prone tree classes; far above non-fire-prone beech (0.075).
    assert 0.300 < nominal_spread_probability(FC.GENERIC_TREE, FC.GENERIC_TREE) < 0.350


@pytest.mark.parametrize("other", sorted(VERIFIED_PROPAGATOR_FUEL_CLASSES, key=lambda f: f.value))
def test_generic_tree_row_and_column_are_the_equal_weight_mixture(other):
    expected_target = (VERIFIED_PROPAGATOR_P_N[CONIFERS][other] + VERIFIED_PROPAGATOR_P_N[BROADLEAVES][other]) / 2
    expected_source = (VERIFIED_PROPAGATOR_P_N[other][CONIFERS] + VERIFIED_PROPAGATOR_P_N[other][BROADLEAVES]) / 2
    assert NOMINAL_SPREAD_PROBABILITY[FC.GENERIC_TREE][other] == pytest.approx(expected_target)
    assert NOMINAL_SPREAD_PROBABILITY[other][FC.GENERIC_TREE] == pytest.approx(expected_source)


def test_methodology_version_marks_the_tree_policy():
    assert METHODOLOGY_VERSION == "1.4"
    assert PROPAGATION_THRESHOLD == 0.45


# --- Behaviour ---------------------------------------------------------------------


@pytest.mark.parametrize("scenario", [ScenarioType.LOW_RISK_NO_FIRE, ScenarioType.MODERATE_RISK_NO_FIRE])
def test_low_and_moderate_weather_never_propagate_tree_fires(scenario):
    for weather in _grid(scenario):
        assert _run(*weather, horizon=30)[0] == [], weather


@pytest.mark.parametrize("scenario", [ScenarioType.HIGH_RISK_NO_FIRE, ScenarioType.ACTIVE_FIRE])
def test_simulator_strong_weather_keeps_generic_tree_fires_risk_only(scenario):
    # Severe simulator corner (41 C, 10% RH, 40 km/h) peaks at p = 0.430 < 0.45: tree fires
    # are reported with real spread-risk cells, but do not propagate in the simulator's range.
    for weather in _grid(scenario):
        spreading, risk_only = _run(*weather, horizon=30)
        assert spreading == [] and len(risk_only) == 8


def test_extreme_realistic_wind_gives_narrow_downwind_only_tree_propagation():
    spreading_30, _ = _run(41.0, 10.0, 60.0, horizon=30)
    spreading_60, _ = _run(41.0, 10.0, 60.0, horizon=60)
    assert (len(spreading_30), len(spreading_60)) == (6, 12)
    assert all(c.latitude < ORIGIN[0] for c in spreading_60)  # wind FROM north: only southward cells


@pytest.mark.parametrize("wind_kmh", [20.0, 30.0, 40.0])
def test_tree_direction_ordering_is_material(wind_kmh):
    moisture = _emc(37.0, 19.0)
    down, cross, up = (
        transition_probability(FC.GENERIC_TREE, FC.GENERIC_TREE, wind_kmh, 0.0, bearing, moisture)
        for bearing in (180.0, 90.0, 0.0)
    )
    assert down > cross > up
    assert down - up > 0.09


def test_built_up_still_maps_to_non_propagating_bare_soil():
    assert _map_dominant_land_cover("Built-up cover") is FC.BARE_SOIL
    cells = FireSpreadCalculator().calculate(
        FireSpreadInput(origin_latitude=ORIGIN[0], origin_longitude=ORIGIN[1], wind_speed_kmh=60.0,
                        wind_direction_deg=0.0, fuel_moisture_percent=0.0, fuel_class=FC.BARE_SOIL, horizon_minutes=60)
    ).cells
    assert all(c.spread_probability < 0.01 for c in cells)


# --- Insufficient data must still exist -------------------------------------------


@pytest.mark.parametrize(
    "label, reason",
    [(None, Reason.MISSING_VEGETATION), ("Moss and lichen cover", Reason.UNSUPPORTED_VEGETATION),
     ("Some New Category", Reason.UNSUPPORTED_VEGETATION)],
)
def test_missing_or_unsupported_vegetation_is_still_insufficient(label, reason):
    service, *_ = build_spread_service(stored_assessment=make_assessment(dominant_land_cover=label))
    result = service.prepare_input(10, AS_OF, horizon_minutes=30)
    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.insufficient_data_reason is reason


# --- Severity -> Spread chain ---------------------------------------------------------


def test_tree_cover_flows_through_severity_and_into_spread():
    # Severity: unchanged methodology still scores Tree cover (fuel score is an input only).
    from tests.services.fire_severity.test_fire_severity_input_service import (
        AS_OF as SEVERITY_AS_OF,
        build_service as build_severity_service,
    )
    from src.external.copernicus import CopernicusCoverFraction, CopernicusLandCoverStatistics

    tree_box = CopernicusLandCoverStatistics(
        cover_fractions=(CopernicusCoverFraction("tree", 0.35), CopernicusCoverFraction("shrub", 0.30),
                         CopernicusCoverFraction("grass", 0.18), CopernicusCoverFraction("built_up", 0.08),
                         CopernicusCoverFraction("crops", 0.07), CopernicusCoverFraction("bare", 0.02)),
        source="COPERNICUS_GLOBAL_LAND_COVER_100M_API", dataset_year=2019, radius_km=1.0,
        sample_count=400, no_data_count=0,
    )
    severity_service, *_ = build_severity_service(statistics=tree_box)
    severity_input = severity_service.prepare_input(10, SEVERITY_AS_OF)
    assert severity_input.vegetation_data.dominant_land_cover == "Tree cover"
    assert severity_input.input_data.vegetation_fuel_score == pytest.approx(0.726)
    assert FireSeverityCalculator().calculate(severity_input.input_data).score > 0

    # Spread: the persisted dominant class now yields a READY input with the generic tree fuel.
    spread_service, *_ = build_spread_service(stored_assessment=make_assessment(dominant_land_cover="Tree cover"))
    spread_input = spread_service.prepare_input(10, AS_OF, horizon_minutes=30)
    assert spread_input.status is FireSpreadInputStatus.READY
    assert spread_input.input_data.fuel_class is FC.GENERIC_TREE
