"""Pure deterministic wildfire-spread cellular-automata calculation.

Implements the verified PROPAGATOR transition equation

    p_ij = (1 - (1 - p_n) ** alpha_wh) * e_m

over a 250 m Moore-neighborhood grid, propagated in fixed 5-minute steps up
to the requested horizon, using EcoGuard V1's deterministic
(non-stochastic) propagation-threshold adaptation instead of the official
model's per-cell random draw. See backend/docs/fire_spread_prediction.md
(Scientific Verification Details) for the full sourcing of every formula and
constant used here.

`PROPAGATION_THRESHOLD` controls only whether a cell can continue
propagating (methodology 1.1, fire_spread_prediction.md §8): neighbours of
spreading cells whose best `p_ij` stays below it are still emitted as
risk-only cells, but never act as propagation sources.

This module is a pure function of its `FireSpreadInput`: no repository
access, no database access, no external API calls, no clock access, and no
random-number generation. Calling `calculate()` twice with identical input
produces exactly equivalent output.

Not implemented here (see backend/docs/fire_spread_prediction.md and the
Task 4B brief for why): terrain/slope (verified neutral at dh=0, so the
official slope term is folded in as an exact constant 1.0 rather than an
input), rate-of-spread-based arrival timing, per-cell vegetation, and any
retrieval of FireEvent/FireSeverityAssessment/WeatherObservation records.
"""
from __future__ import annotations

import math

from src.calculators.fire_spread.fire_spread_config import (
    CA_TIME_STEP_MINUTES,
    CELL_SIZE_METERS,
    METHODOLOGY_NAME,
    METHODOLOGY_VERSION,
    MOISTURE_OF_EXTINCTION,
    MOISTURE_POLYNOMIAL_COEFFICIENTS,
    NOMINAL_SPREAD_PROBABILITY,
    PREDICTION_RADIUS_KM,
    PROPAGATION_THRESHOLD,
    WIND_D1,
    WIND_D2,
    WIND_D3,
    WIND_D4,
    WIND_D5,
    WIND_NEGATIVE_RESCALE,
    WIND_POSITIVE_RESCALE,
    WIND_SPEED_CLIP_KMH,
    WIND_TOPOGRAPHY_A,
)
from src.models.fire_spread_calculation import FireSpreadCalculation
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_prediction import FireSpreadPredictionCell
from src.utils.geo import destination_point

_RADIUS_TOLERANCE_KM = 1e-9

# Moore neighborhood, deterministic order N, NE, E, SE, S, SW, W, NW. Each
# offset is (row_delta, col_delta) in local grid indices (row=+north,
# col=+east) paired with the fixed compass bearing (deg clockwise from
# north) of "source -> target" travel along that offset. Using fixed
# per-offset bearings (rather than recomputing a geodesic bearing for every
# adjacent cell pair) exactly matches how the official PROPAGATOR
# implementation itself operates on its raster grid (a fixed `angle` lookup
# table per neighbor offset in `propagator/constants.py`), and is an
# excellent approximation at this 250 m scale regardless.
_MOORE_OFFSETS: tuple[tuple[tuple[int, int], float], ...] = (
    ((1, 0), 0.0),  # N
    ((1, 1), 45.0),  # NE
    ((0, 1), 90.0),  # E
    ((-1, 1), 135.0),  # SE
    ((-1, 0), 180.0),  # S
    ((-1, -1), 225.0),  # SW
    ((0, -1), 270.0),  # W
    ((1, -1), 315.0),  # NW
)

_ORIGIN_KEY = (0, 0)


class FireSpreadCalculator:
    """Calculate deterministic wildfire-spread prediction cells from normalized input."""

    def calculate(self, input_data: FireSpreadInput) -> FireSpreadCalculation:
        """Return the predicted cells (possibly empty) for one CA run."""
        if not isinstance(input_data, FireSpreadInput):
            raise ValueError(f"input_data must be a FireSpreadInput, got {input_data!r}")

        horizon_steps = input_data.horizon_minutes // CA_TIME_STEP_MINUTES
        grid = _generate_grid_cells(input_data.origin_latitude, input_data.origin_longitude)
        reached_step, reached_probability, risk_only = _run_cellular_automata(
            grid=grid,
            fuel_class=input_data.fuel_class,
            wind_speed_kmh=input_data.wind_speed_kmh,
            wind_direction_deg=input_data.wind_direction_deg,
            fuel_moisture_percent=input_data.fuel_moisture_percent,
            horizon_steps=horizon_steps,
        )
        cells = _build_cells(grid, reached_step, reached_probability, risk_only)

        return FireSpreadCalculation(
            cells=cells,
            horizon_minutes=input_data.horizon_minutes,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
        )


# ---------------------------------------------------------------------------
# Scientific functions: p_n, e_m, alpha_wh, p_ij
# ---------------------------------------------------------------------------


def nominal_spread_probability(
    source_fuel_class: FireSpreadFuelClass,
    target_fuel_class: FireSpreadFuelClass,
) -> float:
    """Return the verified p_n for propagation from a burning to a neighbor cell.

    `source_fuel_class` is the burning/i cell's fuel class, `target_fuel_class`
    is the neighbor/j cell's fuel class. Matches the official source's own
    `prob_table[veg_to - 1, veg_from - 1]` orientation exactly: indexed
    [target][source].
    """
    return NOMINAL_SPREAD_PROBABILITY[target_fuel_class][source_fuel_class]


def moisture_factor(fuel_moisture_percent: float) -> float:
    """Return the verified PROPAGATOR moisture factor e_m for a fuel-moisture percent.

    Reproduces `moist_proba_correction_1` exactly: the percent input is
    converted to a fraction, clipped to [0, 1], divided by the verified
    moisture of extinction (Mx = 0.30), evaluated through the verified
    quintic polynomial, then clipped to [0, 1] -- matching the official
    implementation's own clamping (not an EcoGuard-invented repair).
    """
    moisture_fraction = min(max(fuel_moisture_percent / 100.0, 0.0), 1.0)
    r = moisture_fraction / MOISTURE_OF_EXTINCTION
    c5, c4, c3, c2, c1, c0 = MOISTURE_POLYNOMIAL_COEFFICIENTS
    e_m = (c5 * r**5) + (c4 * r**4) + (c3 * r**3) + (c2 * r**2) + (c1 * r) + c0
    return min(max(e_m, 0.0), 1.0)


def wind_topography_factor(
    wind_speed_kmh: float,
    wind_direction_deg: float,
    propagation_bearing_deg: float,
) -> float:
    """Return the verified alpha_wh for one source-to-target propagation direction.

    `wind_direction_deg` is the meteorological convention (deg clockwise
    from north, the direction FROM which the wind blows).
    `propagation_bearing_deg` is the compass bearing (deg clockwise from
    north) of travel from the source cell toward the target cell.

    Terrain is EcoGuard V1 neutral (dh = 0 always): the verified slope term
    `h_effect = 2 ** tanh((slope * 3) ** 2 * sign(slope))` is exactly 1.0 at
    slope = 0, so it is folded in as a constant rather than exposed as an
    input (see fire_spread_prediction.md §5).
    """
    # km/h, unconverted: the official probability factor takes the model's raw
    # km/h wind (the m/s conversion exists only in its rate-of-spread functions).
    wind_speed = min(max(wind_speed_kmh, 0.0), WIND_SPEED_CLIP_KMH)

    wind_math_angle_rad = _wind_math_angle_rad(wind_direction_deg)
    propagation_math_angle_rad = _propagation_math_angle_rad(propagation_bearing_deg)

    w_effect_module = (
        WIND_TOPOGRAPHY_A
        + WIND_D1 * (WIND_D2 * math.tanh((wind_speed / WIND_D3) - WIND_D4))
        + wind_speed / WIND_D5
    )
    a = (w_effect_module - 1.0) / 4.0
    w_effect_on_direction = (
        (a + 1.0)
        * (1.0 - a**2)
        / (1.0 - a * math.cos(wind_math_angle_rad - propagation_math_angle_rad))
    )
    h_effect = 1.0  # EcoGuard V1 neutral-terrain adaptation; exact at dh = 0.
    combined = h_effect * w_effect_on_direction

    wh = combined - 1.0
    if wh > 0:
        wh /= WIND_POSITIVE_RESCALE
    elif wh < 0:
        wh /= WIND_NEGATIVE_RESCALE
    return wh + 1.0


def transition_probability(
    source_fuel_class: FireSpreadFuelClass,
    target_fuel_class: FireSpreadFuelClass,
    wind_speed_kmh: float,
    wind_direction_deg: float,
    propagation_bearing_deg: float,
    fuel_moisture_percent: float,
) -> float:
    """Return the verified p_ij = (1 - (1 - p_n) ** alpha_wh) * e_m."""
    p_n = nominal_spread_probability(source_fuel_class, target_fuel_class)
    alpha_wh = wind_topography_factor(wind_speed_kmh, wind_direction_deg, propagation_bearing_deg)
    e_m = moisture_factor(fuel_moisture_percent)
    p = 1.0 - (1.0 - p_n) ** alpha_wh
    return min(max(p * e_m, 0.0), 1.0)


def _wind_math_angle_rad(wind_direction_deg: float) -> float:
    """Convert meteorological "from" degrees to the verified internal math angle.

    Reproduces the official implementation's own conversion exactly:
    `wdir = normalize((180 - w_dir_deg + 90) * pi / 180)`, i.e.
    `(270 - w_dir_deg) deg -> rad` (see fire_spread_prediction.md §15.5).
    """
    return math.radians((270.0 - wind_direction_deg) % 360.0)


def _propagation_math_angle_rad(bearing_deg: float) -> float:
    """Convert a compass "toward" bearing into the same math-angle convention
    as `_wind_math_angle_rad`, so the two can be compared via cosine."""
    return math.radians((90.0 - bearing_deg) % 360.0)


# ---------------------------------------------------------------------------
# Grid generation and CA propagation
# ---------------------------------------------------------------------------


def _generate_grid_cells(
    origin_latitude: float,
    origin_longitude: float,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Return {(row, col): (latitude, longitude)} for every cell center within
    PREDICTION_RADIUS_KM of the origin, on a CELL_SIZE_METERS grid centered
    on the origin (row=+north, col=+east). Always includes (0, 0) = origin.
    """
    cell_size_km = CELL_SIZE_METERS / 1000.0
    max_offset = math.ceil(PREDICTION_RADIUS_KM / cell_size_km)

    cells: dict[tuple[int, int], tuple[float, float]] = {}
    for row in range(-max_offset, max_offset + 1):
        for col in range(-max_offset, max_offset + 1):
            east_km = col * cell_size_km
            north_km = row * cell_size_km
            radial_km = math.hypot(east_km, north_km)
            if radial_km > PREDICTION_RADIUS_KM + _RADIUS_TOLERANCE_KM:
                continue
            bearing_deg = math.degrees(math.atan2(east_km, north_km)) % 360.0
            latitude, longitude = destination_point(origin_latitude, origin_longitude, bearing_deg, radial_km)
            cells[(row, col)] = (latitude, longitude)
    return cells


def _run_cellular_automata(
    grid: dict[tuple[int, int], tuple[float, float]],
    fuel_class: FireSpreadFuelClass,
    wind_speed_kmh: float,
    wind_direction_deg: float,
    fuel_moisture_percent: float,
    horizon_steps: int,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], float],
    dict[tuple[int, int], tuple[float, int]],
]:
    """Run the deterministic Moore-neighborhood CA and return per-cell reach state.

    EcoGuard deterministic multi-parent aggregation rule (see
    fire_spread_prediction.md): a target cell may have more than one already
    -reached Moore neighbor evaluated in the same step. All incoming p_ij
    values for that step are computed; the maximum is compared against
    PROPAGATION_THRESHOLD, and -- if the target becomes reached -- that
    maximum is stored as its transition probability. This is a deterministic
    conflict-resolution rule, not a new scientific spread formula.

    A target whose maximum stays below PROPAGATION_THRESHOLD (but above 0)
    is tracked as risk-only: {key: (highest probability, earliest step that
    probability was recorded)}. Risk-only cells are never propagation
    sources, and are dropped from risk-only tracking if a later step makes
    them reached (§8, methodology 1.1).
    """
    reached_step: dict[tuple[int, int], int] = {_ORIGIN_KEY: 0}
    reached_probability: dict[tuple[int, int], float] = {}
    risk_only: dict[tuple[int, int], tuple[float, int]] = {}

    for step in range(1, horizon_steps + 1):
        newly_reached: dict[tuple[int, int], float] = {}
        for target_key in sorted(grid):
            if target_key in reached_step:
                continue

            candidate_probabilities = []
            for (row_delta, col_delta), bearing_deg in _MOORE_OFFSETS:
                source_key = (target_key[0] - row_delta, target_key[1] - col_delta)
                if source_key not in grid or source_key not in reached_step:
                    continue
                candidate_probabilities.append(
                    transition_probability(
                        source_fuel_class=fuel_class,
                        target_fuel_class=fuel_class,
                        wind_speed_kmh=wind_speed_kmh,
                        wind_direction_deg=wind_direction_deg,
                        propagation_bearing_deg=bearing_deg,
                        fuel_moisture_percent=fuel_moisture_percent,
                    )
                )

            if not candidate_probabilities:
                continue
            max_probability = max(candidate_probabilities)
            if max_probability >= PROPAGATION_THRESHOLD:
                newly_reached[target_key] = max_probability
            elif max_probability > 0.0:
                previous = risk_only.get(target_key)
                if previous is None or max_probability > previous[0]:
                    risk_only[target_key] = (max_probability, step)

        for target_key, probability in newly_reached.items():
            reached_step[target_key] = step
            reached_probability[target_key] = probability
            risk_only.pop(target_key, None)

    return reached_step, reached_probability, risk_only


def _build_cells(
    grid: dict[tuple[int, int], tuple[float, float]],
    reached_step: dict[tuple[int, int], int],
    reached_probability: dict[tuple[int, int], float],
    risk_only: dict[tuple[int, int], tuple[float, int]],
) -> tuple[FireSpreadPredictionCell, ...]:
    """Emit spreading cells (p >= PROPAGATION_THRESHOLD) and risk-only cells
    (0 < p < PROPAGATION_THRESHOLD) in one list. For a risk-only cell,
    `reached_step` is the step at which its stored risk was recorded."""
    emitted: list[tuple[tuple[int, int], float, int]] = [
        (key, probability, reached_step[key]) for key, probability in reached_probability.items()
    ]
    emitted.extend((key, probability, step) for key, (probability, step) in risk_only.items())

    cells = []
    for key, probability, step in emitted:
        latitude, longitude = grid[key]
        cells.append(
            FireSpreadPredictionCell(
                latitude=latitude,
                longitude=longitude,
                spread_probability=probability,
                spread_risk_score=probability * 100.0,
                reached_step=step,
                reached_minutes=step * CA_TIME_STEP_MINUTES,
            )
        )
    cells.sort(key=lambda cell: (cell.reached_step, cell.latitude, cell.longitude))
    return tuple(cells)
