"""Independent methodology reference for the PROPAGATOR probability wind factor alpha_wh.

Transcribed from the official CIMAFoundation propagator_sim `legacy` branch
(propagator/constants.py D1..D5, A; propagator/propagator.py `w_h_effect` and
`w_h_effect_on_p`) with LITERAL constants - no production helper or config
constant is imported here, so this is an independent mathematical reference.

Unit: the official model passes its raw input `w_speed` straight into
`w_h_effect_on_p` (probability factor), while the rate-of-spread and spotting
functions of the same file explicitly convert that same variable with
`w_speed / 3.6  # wind speed [m/s]`. The raw model input - and therefore the
probability wind factor's W_s - is km/h. (The ROS wind factor
`exp(0.1783 * V)`, Wang: `beta1 * w_spd` with `w_spd = w_speed * cos / 3.6`,
is the separate m/s formula and is not used by EcoGuard's p_ij.)
"""
from __future__ import annotations

import math

import pytest

from src.calculators.fire_spread.fire_spread_calculator import wind_topography_factor

# Literal official constants (propagator/constants.py).
_D1, _D2, _D3, _D4, _D5 = 0.5, 1.4, 8.2, 2.0, 50.0
_A = 1 - ((_D1 * (_D2 * math.tanh((0 / _D3) - _D4))) + (0 / _D5))
_CLIP_KMH = 60.0  # w_h_effect_on_p: np.clip(w_speed, 0, 60)
_POSITIVE_RESCALE, _NEGATIVE_RESCALE = 2.13, 1.12


def reference_alpha_wh(wind_speed_kmh: float, angle_between_wind_and_spread_rad: float) -> float:
    """Official w_h_effect_on_p with dh = 0 (h_effect = 1); W_s in km/h, no unit conversion."""
    w = min(max(wind_speed_kmh, 0.0), _CLIP_KMH)
    w_effect_module = _A + (_D1 * (_D2 * math.tanh((w / _D3) - _D4))) + (w / _D5)
    a = (w_effect_module - 1) / 4
    w_h = (a + 1) * (1 - a**2) / (1 - a * math.cos(angle_between_wind_and_spread_rad))
    wh = w_h - 1.0
    if wh > 0:
        wh /= _POSITIVE_RESCALE
    elif wh < 0:
        wh /= _NEGATIVE_RESCALE
    return wh + 1.0


# EcoGuard wind direction is meteorological "FROM" 0 deg (north), so the wind
# blows toward 180: downwind bearing 180 (angle 0), crosswind 90 (pi/2), upwind 0 (pi).
_DIRECTIONS = (("downwind", 180.0, 0.0), ("crosswind", 90.0, math.pi / 2), ("upwind", 0.0, math.pi))


def test_reference_constant_a_requires_the_minus_two_inside_tanh():
    # A = 1.67481... is w_effect_module's neutral offset ONLY with D4 inside tanh
    # (tanh(W/8.2 - 2)); the alternative "0.5*(1.4*tanh(W/8.2) - 2)" would give 2.0
    # and a non-neutral factor at zero wind.
    assert _A == pytest.approx(1.67481, abs=1e-5)
    assert 1 - (_D1 * (_D2 * math.tanh(0 / _D3) - _D4)) == pytest.approx(2.0)
    assert reference_alpha_wh(0.0, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("wind_kmh", [0.0, 10.0, 20.0, 27.5, 40.0, 60.0])
@pytest.mark.parametrize("label, bearing_deg, angle_rad", _DIRECTIONS)
def test_implementation_matches_reference_in_kmh(wind_kmh, label, bearing_deg, angle_rad):
    assert wind_topography_factor(wind_kmh, 0.0, bearing_deg) == pytest.approx(
        reference_alpha_wh(wind_kmh, angle_rad), abs=1e-9
    ), label


@pytest.mark.parametrize(
    "wind_kmh, downwind, crosswind, upwind",
    [
        (0.0, 1.0000, 1.0000, 1.0000),
        (20.0, 1.3747, 1.0869, 0.8962),
        (27.5, 1.5303, 1.0711, 0.8116),
        (40.0, 1.6477, 1.0415, 0.7371),
        (60.0, 1.7989, 0.9666, 0.6300),
    ],
)
def test_pinned_reference_values(wind_kmh, downwind, crosswind, upwind):
    assert wind_topography_factor(wind_kmh, 0.0, 180.0) == pytest.approx(downwind, abs=1e-4)
    assert wind_topography_factor(wind_kmh, 0.0, 90.0) == pytest.approx(crosswind, abs=1e-4)
    assert wind_topography_factor(wind_kmh, 0.0, 0.0) == pytest.approx(upwind, abs=1e-4)


def test_wind_above_60_kmh_is_clipped_like_the_official_model():
    assert wind_topography_factor(100.0, 0.0, 180.0) == pytest.approx(wind_topography_factor(60.0, 0.0, 180.0))


@pytest.mark.parametrize("wind_kmh", [20.0, 30.0, 40.0])
def test_direction_is_materially_discriminating_at_simulator_winds(wind_kmh):
    down = wind_topography_factor(wind_kmh, 0.0, 180.0)
    up = wind_topography_factor(wind_kmh, 0.0, 0.0)
    assert down > 1.3 and up < 0.9
