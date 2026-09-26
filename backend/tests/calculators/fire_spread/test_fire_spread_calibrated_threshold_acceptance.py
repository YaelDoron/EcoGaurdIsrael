"""Acceptance of the EcoGuard-calibrated PROPAGATION_THRESHOLD (0.45, Task 11C).

Runs the PRODUCTION calculator and threshold (no patching) over the
simulator's own weather profiles, with fuel moisture from the same FFWI
equilibrium-moisture estimate FireSpreadInputService uses. No location name
enters the calculator - fuel classes are the real Copernicus-mapped classes.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.calculators.fire_spread.fire_spread_calculator import FireSpreadCalculator
from src.models.fire_spread_fuel_class import FireSpreadFuelClass as FC
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_prediction import PROPAGATION_THRESHOLD
from src.services.fire_spread.fire_spread_input_service import _equilibrium_moisture_percent
from src.simulation.generators.weather_data_generator import WEATHER_SCENARIO_PROFILES
from src.simulation.scenario_type import ScenarioType

ORIGIN = (31.782972, 35.136200)
WIND_FROM_NORTH = 0.0  # blows toward the south: south of the origin is downwind
COMBUSTIBLE = (FC.GRASSLAND, FC.SHRUBS, FC.CONIFERS_FIRE_PRONE, FC.AGRO_FORESTRY, FC.BROADLEAVES_FIRE_PRONE)
REPO_ROOT = Path(__file__).resolve().parents[4]


def _moisture(temperature_c: float, relative_humidity: float) -> float:
    return _equilibrium_moisture_percent(SimpleNamespace(temperature=temperature_c, relative_humidity=relative_humidity))


def _run(fuel: FC, temperature_c: float, relative_humidity: float, wind_kmh: float, horizon: int = 30):
    cells = FireSpreadCalculator().calculate(
        FireSpreadInput(
            origin_latitude=ORIGIN[0],
            origin_longitude=ORIGIN[1],
            wind_speed_kmh=wind_kmh,
            wind_direction_deg=WIND_FROM_NORTH,
            fuel_moisture_percent=_moisture(temperature_c, relative_humidity),
            fuel_class=fuel,
            horizon_minutes=horizon,
        )
    ).cells
    spreading = [c for c in cells if c.spread_probability >= PROPAGATION_THRESHOLD]
    risk_only = [c for c in cells if c.spread_probability < PROPAGATION_THRESHOLD]
    return spreading, risk_only


def _grid(scenario: ScenarioType):
    profile = WEATHER_SCENARIO_PROFILES[scenario]
    spans = [
        (low, (low + high) / 2, high)
        for low, high in (profile.temperature_celsius, profile.relative_humidity_percent, profile.wind_speed_kmh)
    ]
    return list(itertools.product(*spans))


def _corner(scenario: ScenarioType, severe: bool):
    p = WEATHER_SCENARIO_PROFILES[scenario]
    if severe:
        return p.temperature_celsius[1], p.relative_humidity_percent[0], p.wind_speed_kmh[1]
    return p.temperature_celsius[0], p.relative_humidity_percent[1], p.wind_speed_kmh[0]


def _upwind(cells):
    return [c for c in cells if c.latitude > ORIGIN[0] + 1e-9]


def _extent_m(cells):
    return max(
        (((c.latitude - ORIGIN[0]) * 111_320.0) ** 2 + ((c.longitude - ORIGIN[1]) * 94_700.0) ** 2) ** 0.5
        for c in cells
    )


def test_production_threshold_is_the_calibrated_value():
    assert PROPAGATION_THRESHOLD == 0.45


def test_frontend_spread_threshold_mirror_matches_the_backend_constant():
    # The UI classifies spreading vs risk-only cells itself (isSpreadingCell); TypeScript
    # cannot import this Python constant, so its mirror literal is pinned here.
    source = (REPO_ROOT / "frontend" / "src" / "types" / "eventDetails.ts").read_text(encoding="utf-8")
    match = re.search(r"export const SPREAD_PROPAGATION_THRESHOLD = ([0-9.]+);", source)
    assert match is not None
    assert float(match.group(1)) == PROPAGATION_THRESHOLD


@pytest.mark.parametrize("fuel", COMBUSTIBLE)
def test_low_risk_weather_on_combustible_fuel_never_propagates(fuel):
    for temperature, humidity, wind in _grid(ScenarioType.LOW_RISK_NO_FIRE):
        spreading, _ = _run(fuel, temperature, humidity, wind)
        assert spreading == [], (fuel, temperature, humidity, wind)


def test_moderate_weather_propagation_is_limited():
    for fuel in (FC.SHRUBS, FC.CONIFERS_FIRE_PRONE, FC.AGRO_FORESTRY, FC.BROADLEAVES_FIRE_PRONE):
        for weather in _grid(ScenarioType.MODERATE_RISK_NO_FIRE):
            assert _run(fuel, *weather)[0] == []
    # Grassland only at the top of the moderate wind range (17 km/h) - 6 of 27 grid points,
    # i.e. the calibrated 4.4% moderate rate over 5 fuels - and only as a narrow downwind finger.
    max_wind = WEATHER_SCENARIO_PROFILES[ScenarioType.MODERATE_RISK_NO_FIRE].wind_speed_kmh[1]
    grass_runs = {weather: _run(FC.GRASSLAND, *weather)[0] for weather in _grid(ScenarioType.MODERATE_RISK_NO_FIRE)}
    propagating = {weather: cells for weather, cells in grass_runs.items() if cells}
    assert len(propagating) == 6
    assert all(wind == max_wind for _, _, wind in propagating)
    for cells in propagating.values():
        assert len(cells) <= 6 and _upwind(cells) == []


def test_strong_grassland_fire_propagates_downwind():
    spreading, _ = _run(FC.GRASSLAND, *_corner(ScenarioType.ACTIVE_FIRE, severe=True))
    assert len(spreading) == 48
    assert _upwind(spreading) == []


def test_strong_shrub_fire_propagates_as_a_narrow_directional_finger():
    at_30, _ = _run(FC.SHRUBS, *_corner(ScenarioType.ACTIVE_FIRE, severe=True), horizon=30)
    at_60, _ = _run(FC.SHRUBS, *_corner(ScenarioType.ACTIVE_FIRE, severe=True), horizon=60)
    assert (len(at_30), len(at_60)) == (6, 12)
    assert _upwind(at_30) == [] and _upwind(at_60) == []
    assert all(abs(c.longitude - ORIGIN[1]) < 1e-9 for c in at_60)  # straight downwind line


def test_bare_or_built_up_fuel_never_meaningfully_propagates():
    spreading, risk_only = _run(FC.BARE_SOIL, 41.0, 10.0, 60.0, horizon=60)
    assert spreading == []
    assert all(c.spread_probability < 0.01 for c in risk_only)


@pytest.mark.parametrize("fuel", [FC.GRASSLAND, FC.SHRUBS])
def test_sixty_minute_extent_exceeds_thirty_minute_extent_when_propagating(fuel):
    weather = _corner(ScenarioType.ACTIVE_FIRE, severe=True)
    at_30, _ = _run(fuel, *weather, horizon=30)
    at_60, _ = _run(fuel, *weather, horizon=60)
    assert _extent_m(at_60) > _extent_m(at_30) * 1.9
    assert max(c.reached_step for c in at_60) == 12 > max(c.reached_step for c in at_30) == 6


def test_typical_active_fire_on_shrubs_stays_risk_only():
    # Not every confirmed fire spreads: mid-range active-fire weather (37 C, 19% RH,
    # 27.5 km/h) on shrubs gives the first ring only, all below the threshold.
    spreading, risk_only = _run(FC.SHRUBS, 37.0, 19.0, 27.5, horizon=60)
    assert spreading == []
    assert len(risk_only) == 8
