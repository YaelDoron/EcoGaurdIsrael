"""Centralized wildfire-spread methodology constants.

Every constant here is labeled per the classification established in
`backend/docs/fire_spread_prediction.md` (Selected Methodology, §3):

- [A] Published PROPAGATOR methodology, verified in Task 4A against the
  official CIMAFoundation `propagator_sim` reference implementation
  (`legacy` branch: `propagator/propagator.py`, `propagator/constants.py`,
  `prob_table.txt`, `v0_table.txt`, `p_vegetation.txt`).
- [C] EcoGuard V1 adaptation -- an EcoGuard implementation decision, not a
  published PROPAGATOR value.

Only methodology constants live here. No database/runtime settings.
"""
from __future__ import annotations

import math

from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_prediction import (
    CA_TIME_STEP_MINUTES,
    PROPAGATION_THRESHOLD,
    SUPPORTED_HORIZON_MINUTES,
)

__all__ = [
    "CA_TIME_STEP_MINUTES",
    "SUPPORTED_HORIZON_MINUTES",
    "CELL_SIZE_METERS",
    "PREDICTION_RADIUS_KM",
    "PROPAGATION_THRESHOLD",
    "MOISTURE_OF_EXTINCTION",
    "MOISTURE_POLYNOMIAL_COEFFICIENTS",
    "WIND_D1",
    "WIND_D2",
    "WIND_D3",
    "WIND_D4",
    "WIND_D5",
    "WIND_TOPOGRAPHY_A",
    "WIND_SPEED_CLIP_KMH",
    "WIND_POSITIVE_RESCALE",
    "WIND_NEGATIVE_RESCALE",
    "NOMINAL_SPREAD_PROBABILITY",
    "VERIFIED_PROPAGATOR_P_N",
    "GENERIC_TREE_SOURCE_CLASSES",
    "METHODOLOGY_NAME",
    "METHODOLOGY_VERSION",
]

# --- Grid / time step [C] (EcoGuard V1 agreed configuration; re-exported here
# from the domain model so the calculator has one centralized import surface,
# per fire_spread_prediction.md §5) --------------------------------------
CELL_SIZE_METERS = 250.0
PREDICTION_RADIUS_KM = 5.0

# --- Deterministic propagation threshold [C] -------------------------------
# Re-exported from src.models.fire_spread_prediction (the single source of
# truth). EcoGuard-calibrated value (0.45, Task 11C) - the official model's
# per-cell ignition test is a stochastic draw (p_prob > rand(...)), so this
# is NOT a PROPAGATOR scientific constant (see fire_spread_prediction.md §3.2, §8).

# --- Moisture factor e_m [A] -------------------------------------------
# Verified from `propagator/propagator.py`, `moist_proba_correction_1` (the
# model's default moisture function), whose own docstring cites
# "Trucchia et al, Fire 2020". Mx is a fraction (0.30 == 30%).
MOISTURE_OF_EXTINCTION = 0.30
# Coefficients in descending power order: r^5, r^4, r^3, r^2, r^1, r^0.
MOISTURE_POLYNOMIAL_COEFFICIENTS = (-11.507, 22.963, -17.331, 6.598, -1.7211, 1.0003)

# --- Wind/topography factor alpha_wh [A] --------------------------------
# Verified from `propagator/constants.py` and `propagator/propagator.py`
# (`w_h_effect`, `w_h_effect_on_p`).
WIND_D1 = 0.5
WIND_D2 = 1.4
WIND_D3 = 8.2
WIND_D4 = 2.0
WIND_D5 = 50.0
# A is `w_effect_module` evaluated at wind speed 0 (verified formula
# construction ensures the wind factor is exactly neutral at zero wind).
WIND_TOPOGRAPHY_A = 1.0 - (WIND_D1 * (WIND_D2 * math.tanh(-WIND_D4)))
# Verified clip: `np.clip(w_speed, 0, 60)` in `w_h_effect_on_p`.
#
# UNIT (resolved, Task 12 - fire_spread_prediction.md §15.3): the official
# model's raw `w_speed` input is **km/h** and is passed to the probability
# factor `w_h_effect_on_p` UNCONVERTED. The same file's rate-of-spread and
# spotting functions convert that same variable explicitly
# (`w_speed / 3.6  # wind speed [m/s]` in `fire_spotting`; `/ 3.6` in
# `p_time_rothermel`/`p_time_wang`, whose Wang wind factor is
# exp(0.1783 * V[m/s])). Those m/s conversions belong to ROS timing only.
# EcoGuard previously applied `/ 3.6` here too, understating wind 3.6x.
WIND_SPEED_CLIP_KMH = 60.0
# Verified probability-rescaling divisors from `w_h_effect_on_p`.
WIND_POSITIVE_RESCALE = 2.13
WIND_NEGATIVE_RESCALE = 1.12

# --- Nominal vegetation spread probability p_n [A] ----------------------
# Verified verbatim from the official `prob_table.txt` (7x7), keyed
# [target_fuel_class][source_fuel_class] to match `prob_table[veg_to - 1,
# veg_from - 1]` in the official source exactly. Includes the Task 4A
# corrected entry: CONIFERS_FIRE_PRONE -> GRASSLAND is 0.100 (not the 0.250
# that had circulated in an earlier, unverified draft).
_BROADLEAVES_FIRE_PRONE = FireSpreadFuelClass.BROADLEAVES_FIRE_PRONE
_SHRUBS = FireSpreadFuelClass.SHRUBS
_BARE_SOIL = FireSpreadFuelClass.BARE_SOIL
_GRASSLAND = FireSpreadFuelClass.GRASSLAND
_CONIFERS_FIRE_PRONE = FireSpreadFuelClass.CONIFERS_FIRE_PRONE
_AGRO_FORESTRY = FireSpreadFuelClass.AGRO_FORESTRY
_BROADLEAVES_NON_FIRE_PRONE = FireSpreadFuelClass.BROADLEAVES_NON_FIRE_PRONE

VERIFIED_PROPAGATOR_P_N: dict[FireSpreadFuelClass, dict[FireSpreadFuelClass, float]] = {
    _BROADLEAVES_FIRE_PRONE: {
        _BROADLEAVES_FIRE_PRONE: 0.300,
        _SHRUBS: 0.375,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.250,
        _CONIFERS_FIRE_PRONE: 0.275,
        _AGRO_FORESTRY: 0.250,
        _BROADLEAVES_NON_FIRE_PRONE: 0.250,
    },
    _SHRUBS: {
        _BROADLEAVES_FIRE_PRONE: 0.375,
        _SHRUBS: 0.375,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.350,
        _CONIFERS_FIRE_PRONE: 0.400,
        _AGRO_FORESTRY: 0.300,
        _BROADLEAVES_NON_FIRE_PRONE: 0.375,
    },
    _BARE_SOIL: {
        _BROADLEAVES_FIRE_PRONE: 0.005,
        _SHRUBS: 0.005,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.005,
        _CONIFERS_FIRE_PRONE: 0.005,
        _AGRO_FORESTRY: 0.005,
        _BROADLEAVES_NON_FIRE_PRONE: 0.005,
    },
    _GRASSLAND: {
        _BROADLEAVES_FIRE_PRONE: 0.450,
        _SHRUBS: 0.475,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.475,
        _CONIFERS_FIRE_PRONE: 0.475,
        _AGRO_FORESTRY: 0.375,
        _BROADLEAVES_NON_FIRE_PRONE: 0.475,
    },
    _CONIFERS_FIRE_PRONE: {
        _BROADLEAVES_FIRE_PRONE: 0.225,
        _SHRUBS: 0.325,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.100,  # Task 4A corrected value (verified official source).
        _CONIFERS_FIRE_PRONE: 0.350,
        _AGRO_FORESTRY: 0.200,
        _BROADLEAVES_NON_FIRE_PRONE: 0.350,
    },
    _AGRO_FORESTRY: {
        _BROADLEAVES_FIRE_PRONE: 0.250,
        _SHRUBS: 0.250,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.300,
        _CONIFERS_FIRE_PRONE: 0.475,
        _AGRO_FORESTRY: 0.350,
        _BROADLEAVES_NON_FIRE_PRONE: 0.250,
    },
    _BROADLEAVES_NON_FIRE_PRONE: {
        _BROADLEAVES_FIRE_PRONE: 0.075,
        _SHRUBS: 0.100,
        _BARE_SOIL: 0.005,
        _GRASSLAND: 0.075,
        _CONIFERS_FIRE_PRONE: 0.275,
        _AGRO_FORESTRY: 0.075,
        _BROADLEAVES_NON_FIRE_PRONE: 0.075,
    },
}

# --- Generic tree fuel [C] (Task 14) ------------------------------------
# Copernicus "Tree cover" carries no leaf type or fire-proneness, so mapping it
# to one PROPAGATOR tree class would assert a species. EcoGuard instead uses an
# equal-weight mixture of the two FIRE-PRONE tree classes, derived entirely
# from the verified table above (no new free parameter):
#   p[GT][x] = mean(p[CONIFERS][x], p[BROADLEAVES_FP][x])     (GT as target)
#   p[x][GT] = mean(p[x][CONIFERS], p[x][BROADLEAVES_FP])     (GT as source)
#   p[GT][GT] = mean(p[CONIFERS][CONIFERS], p[BROADLEAVES_FP][BROADLEAVES_FP]) = 0.325
# (the diagonal uses the same-class pairing, consistent with the homogeneous
# fuel grid). PROPAGATOR's non-fire-prone broadleaves class ("faggete", beech
# woods) is excluded as a documented domain assumption: beech forest does not
# occur in Israel's Mediterranean flora. See fire_spread_prediction.md §4.3.2.
GENERIC_TREE_SOURCE_CLASSES = (_CONIFERS_FIRE_PRONE, _BROADLEAVES_FIRE_PRONE)


def _with_generic_tree(
    verified: dict[FireSpreadFuelClass, dict[FireSpreadFuelClass, float]],
) -> dict[FireSpreadFuelClass, dict[FireSpreadFuelClass, float]]:
    generic = FireSpreadFuelClass.GENERIC_TREE
    first, second = GENERIC_TREE_SOURCE_CLASSES
    table = {target: dict(row) for target, row in verified.items()}
    for target in verified:
        table[target][generic] = (verified[target][first] + verified[target][second]) / 2.0
    table[generic] = {source: (verified[first][source] + verified[second][source]) / 2.0 for source in verified}
    table[generic][generic] = (verified[first][first] + verified[second][second]) / 2.0
    return table


# Effective table used by the calculator: the verified PROPAGATOR values,
# unchanged, plus the derived GENERIC_TREE row/column.
NOMINAL_SPREAD_PROBABILITY = _with_generic_tree(VERIFIED_PROPAGATOR_P_N)

# --- Methodology identity [C] -------------------------------------------
METHODOLOGY_NAME = "ECOGUARD_PROPAGATOR_CA"
# 1.1: sub-threshold neighbours are emitted as risk-only cells (§8).
# 1.2: probability wind factor takes km/h unconverted, as the official model does (§15.3).
# 1.3: calibrated PROPAGATION_THRESHOLD 0.50 -> 0.45 (§8). The effective-state
#      fingerprint covers the version but not the threshold, so this bump is what
#      keeps 0.50-era predictions from being reused as NO_OP.
# 1.4: Copernicus "Tree cover" -> derived GENERIC_TREE fuel instead of
#      INSUFFICIENT_DATA (§4.3.2).
METHODOLOGY_VERSION = "1.4"
