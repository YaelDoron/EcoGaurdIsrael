"""Deterministic calibration matrix for the fire-spread PROPAGATION_THRESHOLD.

Analysis tool only: never imported by the app, never writes anything, and
NEVER changes the production threshold. Candidate thresholds are evaluated by
temporarily patching the calculator module's constant inside a context
manager (the same technique the existing calculator tests use), and every
probability comes from the real `transition_probability` / `moisture_factor`
/ `wind_topography_factor` functions. Weather comes from the simulator's own
WEATHER_SCENARIO_PROFILES; fuel moisture from the same FFWI equilibrium
moisture helpers FireSpreadInputService uses.

Usage (from backend/):
    python -m scripts.calibrate_fire_spread_threshold
"""
from __future__ import annotations

import itertools
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from unittest import mock

from src.calculators.fire_danger.ffwi_calculator import _celsius_to_fahrenheit, _equilibrium_moisture_content
from src.calculators.fire_spread import fire_spread_calculator as fsc
from src.calculators.fire_spread.fire_spread_calculator import FireSpreadCalculator, moisture_factor, transition_probability
from src.calculators.fire_spread.fire_spread_config import PROPAGATION_THRESHOLD
from src.models.fire_spread_fuel_class import FireSpreadFuelClass as FC
from src.models.fire_spread_input import FireSpreadInput
from src.simulation.generators.weather_data_generator import WEATHER_SCENARIO_PROFILES
from src.simulation.scenario_type import ScenarioType

# Defined from the score distribution (burnable same-fuel p_ij spans ~0.21-0.47 under
# simulator weather) BEFORE any location scenario was compared.
CANDIDATE_THRESHOLDS: tuple[float, ...] = (0.30, 0.325, 0.35, 0.375, 0.40, 0.45, 0.50)

REGIMES: tuple[tuple[str, ScenarioType], ...] = (
    ("low", ScenarioType.LOW_RISK_NO_FIRE),
    ("moderate", ScenarioType.MODERATE_RISK_NO_FIRE),
    ("high", ScenarioType.HIGH_RISK_NO_FIRE),
    ("active_fire", ScenarioType.ACTIVE_FIRE),
)

# Fuels reachable from EcoGuard's Copernicus mapping (fire_spread_input_service
# _DOMINANT_LAND_COVER_TO_FUEL_CLASS) vs. calculator-only classes.
MAPPED_FUELS = (FC.GRASSLAND, FC.SHRUBS, FC.AGRO_FORESTRY, FC.BARE_SOIL)
BURNABLE_FUELS = (
    FC.GRASSLAND,
    FC.SHRUBS,
    FC.CONIFERS_FIRE_PRONE,
    FC.AGRO_FORESTRY,
    FC.BROADLEAVES_FIRE_PRONE,
)

# Relative to the wind: 0 = straight downwind, 90 = crosswind, 180 = straight upwind.
DIRECTIONS: tuple[tuple[str, float], ...] = (("downwind", 0.0), ("crosswind", 90.0), ("upwind", 180.0))
_WIND_FROM_DEG = 0.0  # wind FROM north -> downwind bearing is 180 (south)

# Real Copernicus Global Land Cover 100 m (2019) dominant classes at each canonical
# simulation location (+/-1 km box, 400 pixels after the Task 13 sampling fix),
# mapped by FireSpreadInputService._map_dominant_land_cover. Galilee (Har Kamon)
# is Tree-dominant -> the EcoGuard-derived GENERIC_TREE fuel (Task 14).
LOCATION_FUEL_CLASSES: dict[str, FC] = {
    "galilee": FC.GENERIC_TREE,  # Tree 35 / Shrub 30 / Grass 18 / Built-up 8 / Crop 7 / Bare 2
    "jerusalem_forest": FC.SHRUBS,  # Shrub 32 / Grass 28 / Tree 27 / Built-up 6 / Crop 4 / Bare 3
    "carmel": FC.SHRUBS,  # Shrub 28 / Tree 24 / Built-up 22 / Grass 16 / Crop 8 / Bare 2
    "judean_hills": FC.GRASSLAND,  # Grass 40 / Shrub 19 / Crop 18 / Tree 15 / Bare 5 / Built-up 2
    "golan": FC.AGRO_FORESTRY,  # Crop 67 / Grass 14 / Shrub 9 / Tree 5 / Built-up 3 / Bare 2
}

# Observed Jerusalem Forest event (FireEvent 726) inputs, see scripts/audit_fire_spread.py.
JERUSALEM_OBSERVED = dict(temperature_c=36.6, relative_humidity=10.9, wind_speed_kmh=27.5)


@dataclass(frozen=True)
class WeatherPoint:
    regime: str
    temperature_c: float
    relative_humidity: float
    wind_speed_kmh: float

    @property
    def fuel_moisture_percent(self) -> float:
        return fuel_moisture_percent(self.temperature_c, self.relative_humidity)


def fuel_moisture_percent(temperature_c: float, relative_humidity: float) -> float:
    """Same estimate FireSpreadInputService._equilibrium_moisture_percent uses."""
    return _equilibrium_moisture_content(
        relative_humidity_pct=relative_humidity,
        temperature_f=_celsius_to_fahrenheit(temperature_c),
    )


def _span(bounds: tuple[float, float]) -> tuple[float, float, float]:
    low, high = bounds
    return (low, (low + high) / 2.0, high)


def regime_weather_grid(regime: str) -> tuple[WeatherPoint, ...]:
    """3x3x3 grid (min/mid/max of temperature, RH and wind) of one simulator profile."""
    profile = WEATHER_SCENARIO_PROFILES[dict(REGIMES)[regime]]
    return tuple(
        WeatherPoint(regime, t, h, w)
        for t, h, w in itertools.product(
            _span(profile.temperature_celsius),
            _span(profile.relative_humidity_percent),
            _span(profile.wind_speed_kmh),
        )
    )


def severe_corner(regime: str) -> WeatherPoint:
    profile = WEATHER_SCENARIO_PROFILES[dict(REGIMES)[regime]]
    return WeatherPoint(
        regime,
        profile.temperature_celsius[1],
        profile.relative_humidity_percent[0],
        profile.wind_speed_kmh[1],
    )


def mild_corner(regime: str) -> WeatherPoint:
    profile = WEATHER_SCENARIO_PROFILES[dict(REGIMES)[regime]]
    return WeatherPoint(
        regime,
        profile.temperature_celsius[0],
        profile.relative_humidity_percent[1],
        profile.wind_speed_kmh[0],
    )


def directional_probability(fuel: FC, weather: WeatherPoint, relative_direction_deg: float) -> float:
    bearing = (_WIND_FROM_DEG + 180.0 + relative_direction_deg) % 360.0
    return transition_probability(
        source_fuel_class=fuel,
        target_fuel_class=fuel,
        wind_speed_kmh=weather.wind_speed_kmh,
        wind_direction_deg=_WIND_FROM_DEG,
        propagation_bearing_deg=bearing,
        fuel_moisture_percent=weather.fuel_moisture_percent,
    )


def reproduce_p_ij(p_n: float, alpha_wh: float, e_m: float) -> float:
    """p_ij = (1 - (1 - p_n) ** alpha_wh) * e_m, clipped to [0, 1]."""
    return min(max((1.0 - (1.0 - p_n) ** alpha_wh) * e_m, 0.0), 1.0)


@contextmanager
def patched_threshold(threshold: float) -> Iterator[None]:
    """Evaluate a CANDIDATE threshold without touching the production constant."""
    with mock.patch.object(fsc, "PROPAGATION_THRESHOLD", threshold):
        yield


@dataclass(frozen=True)
class CaOutcome:
    propagated_cells: int
    risk_only_cells: int
    max_reached_step: int
    max_distance_m: float
    downwind_cells: int
    upwind_cells: int


def run_ca(fuel: FC, weather: WeatherPoint, horizon_minutes: int, threshold: float) -> CaOutcome:
    origin_lat, origin_lon = 31.782972, 35.136200
    input_data = FireSpreadInput(
        origin_latitude=origin_lat,
        origin_longitude=origin_lon,
        wind_speed_kmh=weather.wind_speed_kmh,
        wind_direction_deg=_WIND_FROM_DEG,
        fuel_moisture_percent=weather.fuel_moisture_percent,
        fuel_class=fuel,
        horizon_minutes=horizon_minutes,
    )
    with patched_threshold(threshold):
        cells = FireSpreadCalculator().calculate(input_data).cells
    propagated = [c for c in cells if c.spread_probability >= threshold]
    downwind = sum(1 for c in propagated if c.latitude < origin_lat - 1e-9)  # wind FROM north -> south is downwind
    upwind = sum(1 for c in propagated if c.latitude > origin_lat + 1e-9)
    distance = max(
        (
            math.hypot(
                (c.latitude - origin_lat) * 111_320.0,
                (c.longitude - origin_lon) * 111_320.0 * math.cos(math.radians(origin_lat)),
            )
            for c in propagated
        ),
        default=0.0,
    )
    return CaOutcome(
        propagated_cells=len(propagated),
        risk_only_cells=len(cells) - len(propagated),
        max_reached_step=max((c.reached_step for c in propagated), default=0),
        max_distance_m=distance,
        downwind_cells=downwind,
        upwind_cells=upwind,
    )


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    low_false_propagation: float  # any direction, burnable fuels, LOW grid
    moderate_downwind_rate: float
    strong_downwind_rate: float  # HIGH + ACTIVE_FIRE grids, burnable fuels
    strong_upwind_rate: float
    directional_consistency: float | None  # of strong downwind-propagating cases, share where upwind does NOT (None: none propagate)
    fuels_propagating_under_strong: tuple[str, ...]


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def threshold_metrics(threshold: float) -> ThresholdMetrics:
    low = [
        directional_probability(f, w, d) >= threshold
        for f in BURNABLE_FUELS
        for w in regime_weather_grid("low")
        for _, d in DIRECTIONS
    ]
    moderate = [
        directional_probability(f, w, 0.0) >= threshold
        for f in BURNABLE_FUELS
        for w in regime_weather_grid("moderate")
    ]
    strong_pairs = [
        (directional_probability(f, w, 0.0) >= threshold, directional_probability(f, w, 180.0) >= threshold, f)
        for f in BURNABLE_FUELS
        for regime in ("high", "active_fire")
        for w in regime_weather_grid(regime)
    ]
    propagating = [pair for pair in strong_pairs if pair[0]]
    return ThresholdMetrics(
        threshold=threshold,
        low_false_propagation=_rate(low),
        moderate_downwind_rate=_rate(moderate),
        strong_downwind_rate=_rate([p[0] for p in strong_pairs]),
        strong_upwind_rate=_rate([p[1] for p in strong_pairs]),
        directional_consistency=_rate([not p[1] for p in propagating]) if propagating else None,
        fuels_propagating_under_strong=tuple(sorted({p[2].name for p in propagating})),
    )


def minimum_downwind_wind_kmh(fuel: FC, fuel_moisture: float, threshold: float, max_kmh: int = 216) -> int | None:
    """Smallest integer wind (km/h) at which the straight-downwind neighbour reaches `threshold`."""
    for wind in range(0, max_kmh + 1):
        p = transition_probability(fuel, fuel, float(wind), _WIND_FROM_DEG, 180.0, fuel_moisture)
        if p >= threshold:
            return wind
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report() -> None:
    print(f"Production PROPAGATION_THRESHOLD = {PROPAGATION_THRESHOLD} (unchanged)\n")

    print("== Simulator weather regimes (WEATHER_SCENARIO_PROFILES) -> fuel moisture / e_m ==")
    for regime, scenario in REGIMES:
        profile = WEATHER_SCENARIO_PROFILES[scenario]
        moistures = [w.fuel_moisture_percent for w in regime_weather_grid(regime)]
        print(
            f"{regime:12} T={profile.temperature_celsius} RH={profile.relative_humidity_percent}"
            f" wind={profile.wind_speed_kmh} km/h -> EMC {min(moistures):.2f}-{max(moistures):.2f}%"
            f" e_m {moisture_factor(max(moistures)):.3f}-{moisture_factor(min(moistures)):.3f}"
        )

    print("\n== p_ij range by fuel over all simulator regimes (min .. max, any direction) ==")
    for fuel in FC:
        values = [
            directional_probability(fuel, w, d)
            for regime, _ in REGIMES
            for w in regime_weather_grid(regime)
            for _, d in DIRECTIONS
        ]
        print(f"{fuel.name:28} {min(values):.3f} .. {max(values):.3f}")

    print("\n== Calibration matrix: p_ij (downwind / crosswind / upwind) ==")
    header = f"{'fuel':24} {'weather':30} {'EMC%':>5} {'down':>6} {'cross':>6} {'up':>6}"
    print(header)
    print("-" * len(header))
    for fuel in (*BURNABLE_FUELS, FC.BARE_SOIL):
        for label, weather in (
            ("low mild", mild_corner("low")),
            ("low severe", severe_corner("low")),
            ("moderate severe", severe_corner("moderate")),
            ("active mild", mild_corner("active_fire")),
            ("active severe", severe_corner("active_fire")),
        ):
            description = f"{label} {weather.temperature_c:.0f}C/{weather.relative_humidity:.0f}%/{weather.wind_speed_kmh:.0f}"
            p = [directional_probability(fuel, weather, d) for _, d in DIRECTIONS]
            print(f"{fuel.name:24} {description:30} {weather.fuel_moisture_percent:5.2f} {p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f}")

    print("\n== Minimum straight-downwind wind (km/h) to reach the threshold ==")
    for moisture_label, moisture in (("EMC 2.26% (driest sim)", 2.26), ("EMC 5.0%", 5.0), ("EMC 8.2% (wettest LOW)", 8.21)):
        print(moisture_label)
        for fuel in BURNABLE_FUELS:
            row = []
            for threshold in CANDIDATE_THRESHOLDS:
                wind = minimum_downwind_wind_kmh(fuel, moisture, threshold)
                row.append(f"{threshold}:{'never' if wind is None else wind}")
            print(f"   {fuel.name:24} " + "  ".join(row))

    print("\n== Candidate thresholds (burnable fuels, simulator grids) ==")
    print(f"{'thr':>6} {'LOW false':>9} {'MOD down':>8} {'STRONG down':>11} {'STRONG up':>9} {'dir.consist':>11}  fuels propagating (strong)")
    for threshold in CANDIDATE_THRESHOLDS:
        m = threshold_metrics(threshold)
        print(
            f"{threshold:6.3f} {m.low_false_propagation:9.1%} {m.moderate_downwind_rate:8.1%}"
            f" {m.strong_downwind_rate:11.1%} {m.strong_upwind_rate:9.1%}"
            f" {'n/a' if m.directional_consistency is None else format(m.directional_consistency, '.1%'):>11}"
            f"  {', '.join(m.fuels_propagating_under_strong) or '-'}"
        )

    print("\n== Location scenarios (real Copernicus fuel) - CA propagated cells 30/60 min ==")
    scenarios = (
        ("low mild (should NOT spread)", mild_corner("low")),
        ("moderate severe (limited)", severe_corner("moderate")),
        ("active typical mid", regime_weather_grid("active_fire")[13]),
        ("observed Jerusalem 36.6C/10.9%/27.5", WeatherPoint("observed", **JERUSALEM_OBSERVED)),
        ("active severe (should spread)", severe_corner("active_fire")),
    )
    for location, fuel in LOCATION_FUEL_CLASSES.items():
        print(f"{location} ({fuel.name})")
        for label, weather in scenarios:
            cells = []
            for threshold in CANDIDATE_THRESHOLDS:
                o30 = run_ca(fuel, weather, 30, threshold)
                o60 = run_ca(fuel, weather, 60, threshold)
                cells.append(f"{threshold}:{o30.propagated_cells}/{o60.propagated_cells}")
            print(f"   {label:38} " + "  ".join(cells))

    print("\n== 30 vs 60 min and direction at candidate thresholds (active severe corner) ==")
    for fuel in (FC.GRASSLAND, FC.SHRUBS, FC.AGRO_FORESTRY):
        for threshold in CANDIDATE_THRESHOLDS:
            o30 = run_ca(fuel, severe_corner("active_fire"), 30, threshold)
            o60 = run_ca(fuel, severe_corner("active_fire"), 60, threshold)
            print(
                f"{fuel.name:14} thr={threshold:5.3f}  30m: {o30.propagated_cells:4d} cells, reach {o30.max_distance_m:5.0f} m,"
                f" down/up {o30.downwind_cells}/{o30.upwind_cells}   60m: {o60.propagated_cells:4d} cells,"
                f" reach {o60.max_distance_m:5.0f} m, down/up {o60.downwind_cells}/{o60.upwind_cells}"
            )


if __name__ == "__main__":
    _print_report()
