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
from src.models.fire_spread_prediction import CA_TIME_STEP_MINUTES, SUPPORTED_HORIZON_MINUTES

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
    "WIND_SPEED_CLIP_MS",
    "WIND_POSITIVE_RESCALE",
    "WIND_NEGATIVE_RESCALE",
    "NOMINAL_SPREAD_PROBABILITY",
    "METHODOLOGY_NAME",
    "METHODOLOGY_VERSION",
]

# --- Grid / time step [C] (EcoGuard V1 agreed configuration; re-exported here
# from the domain model so the calculator has one centralized import surface,
# per fire_spread_prediction.md §5) --------------------------------------
CELL_SIZE_METERS = 250.0
PREDICTION_RADIUS_KM = 5.0

# --- Deterministic propagation threshold [C] -------------------------------
# EcoGuard-specific adaptation. The official model's per-cell ignition test
# is a stochastic draw (p_prob > rand(...)); 0.5 is NOT a PROPAGATOR
# scientific constant (see fire_spread_prediction.md §3.2, §8).
PROPAGATION_THRESHOLD = 0.5

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
# A is `w_effect_module` evaluated at wind_speed_ms = 0 (verified formula
# construction ensures the wind factor is exactly neutral at zero wind).
WIND_TOPOGRAPHY_A = 1.0 - (WIND_D1 * (WIND_D2 * math.tanh(-WIND_D4)))
# Verified clip: `np.clip(w_speed, 0, 60)` in `w_h_effect_on_p`.
#
# OPEN ITEM (fire_spread_prediction.md §15.3, §15.7 item 2): the official
# source does not state whether this `w_speed` is expected in km/h or m/s.
# This EcoGuard implementation assumes **m/s**, reasoned from the D3=8.2
# tanh-saturation midpoint being physically implausible as a km/h value
# (an 8.2 km/h "half-effect" wind would saturate the wind factor for nearly
# all real fire-weather conditions) and far more plausible as an ~8.2 m/s
# (~30 km/h) saturation point. This is a flagged, overridable decision, not
# a silently invented one -- see fire_spread_prediction.md for the full
# reasoning. `wind_speed_kmh` inputs are converted via `wind_speed_kmh / 3.6`
# (exact unit math, matching the official RoS functions' own conversion)
# before this clip and the tanh formula are applied.
WIND_SPEED_CLIP_MS = 60.0
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

NOMINAL_SPREAD_PROBABILITY: dict[FireSpreadFuelClass, dict[FireSpreadFuelClass, float]] = {
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

# --- Methodology identity [C] -------------------------------------------
METHODOLOGY_NAME = "ECOGUARD_PROPAGATOR_CA"
# 1.1: sub-threshold neighbours are emitted as risk-only cells (§8).
METHODOLOGY_VERSION = "1.1"
