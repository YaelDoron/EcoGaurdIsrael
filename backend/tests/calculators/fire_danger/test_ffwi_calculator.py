"""Tests for the Fosberg Fire Weather Index calculator."""
from __future__ import annotations

import math

import pytest

from src.calculators.fire_danger.ffwi_calculator import FFWICalculator, _classify_score
from src.calculators.fire_danger.ffwi_config import FFWI_MAX_SCORE, FFWI_MIN_SCORE
from src.models import FireDangerInput, FireDangerLevel


def expected_ffwi_score(temperature_c: float, relative_humidity_pct: float, wind_speed_kmh: float) -> float:
    temperature_f = temperature_c * 9 / 5 + 32
    wind_speed_mph = wind_speed_kmh / 1.609344
    h = relative_humidity_pct
    if h < 10:
        moisture = 0.03229 + 0.281073 * h - 0.000578 * h * temperature_f
    elif h <= 50:
        moisture = 2.22749 + 0.160107 * h - 0.01478 * temperature_f
    else:
        moisture = 21.0606 + 0.005565 * h**2 - 0.00035 * h * temperature_f - 0.483199 * h
    x = moisture / 30
    eta = 1 - 2 * x + 1.5 * x**2 - 0.5 * x**3
    raw_score = eta * math.sqrt(1 + wind_speed_mph**2) / 0.3002
    return min(max(raw_score, FFWI_MIN_SCORE), FFWI_MAX_SCORE)


def calculate(temperature_c: float, relative_humidity_pct: float, wind_speed_kmh: float):
    return FFWICalculator().calculate(
        FireDangerInput(
            temperature_c=temperature_c,
            relative_humidity_pct=relative_humidity_pct,
            wind_speed_kmh=wind_speed_kmh,
        )
    )


def test_hot_dry_windy_conditions_score_higher_than_cool_humid_calm_conditions():
    hot_dry_windy = calculate(temperature_c=40, relative_humidity_pct=8, wind_speed_kmh=45)
    cool_humid_calm = calculate(temperature_c=15, relative_humidity_pct=85, wind_speed_kmh=2)

    assert hot_dry_windy.score > cool_humid_calm.score


def test_identical_input_produces_identical_output_repeatedly():
    input_data = FireDangerInput(temperature_c=31.5, relative_humidity_pct=22.0, wind_speed_kmh=18.0)
    calculator = FFWICalculator()

    first = calculator.calculate(input_data)
    second = calculator.calculate(input_data)

    assert first == second


@pytest.mark.parametrize(
    ("temperature_c", "relative_humidity_pct", "wind_speed_kmh"),
    [
        (0, 0, 0),
        (20, 10, 5),
        (33, 50, 30),
        (45, 100, 80),
        (60, 0, 5000),
    ],
)
def test_result_score_is_always_between_zero_and_one_hundred(
    temperature_c,
    relative_humidity_pct,
    wind_speed_kmh,
):
    result = calculate(temperature_c, relative_humidity_pct, wind_speed_kmh)

    assert FFWI_MIN_SCORE <= result.score <= FFWI_MAX_SCORE


@pytest.mark.parametrize(
    ("relative_humidity_pct", "expected_moisture"),
    [
        (9.999, 0.03229 + 0.281073 * 9.999 - 0.000578 * 9.999 * 68.0),
        (10.0, 2.22749 + 0.160107 * 10.0 - 0.01478 * 68.0),
        (30.0, 2.22749 + 0.160107 * 30.0 - 0.01478 * 68.0),
        (50.0, 2.22749 + 0.160107 * 50.0 - 0.01478 * 68.0),
        (50.001, 21.0606 + 0.005565 * 50.001**2 - 0.00035 * 50.001 * 68.0 - 0.483199 * 50.001),
    ],
)
def test_moisture_equation_branches_use_required_boundaries(relative_humidity_pct, expected_moisture):
    x = expected_moisture / 30
    eta = 1 - 2 * x + 1.5 * x**2 - 0.5 * x**3
    expected_score = min(max(eta * math.sqrt(1 + 0**2) / 0.3002, FFWI_MIN_SCORE), FFWI_MAX_SCORE)

    result = calculate(temperature_c=20.0, relative_humidity_pct=relative_humidity_pct, wind_speed_kmh=0.0)

    assert result.score == pytest.approx(expected_score)
    assert result.score == pytest.approx(
        expected_ffwi_score(20.0, relative_humidity_pct, 0.0)
    )


@pytest.mark.parametrize(
    ("temperature_c", "relative_humidity_pct", "wind_speed_kmh"),
    [
        (25, 30, 0),
        (25, 0, 10),
        (25, 100, 10),
    ],
)
def test_boundary_valid_inputs_are_accepted(temperature_c, relative_humidity_pct, wind_speed_kmh):
    result = calculate(temperature_c, relative_humidity_pct, wind_speed_kmh)

    assert FFWI_MIN_SCORE <= result.score <= FFWI_MAX_SCORE


@pytest.mark.parametrize(
    "input_data",
    [
        dict(temperature_c=25, relative_humidity_pct=30, wind_speed_kmh=-0.1),
        dict(temperature_c=25, relative_humidity_pct=-0.1, wind_speed_kmh=10),
        dict(temperature_c=25, relative_humidity_pct=100.1, wind_speed_kmh=10),
        dict(temperature_c=math.nan, relative_humidity_pct=30, wind_speed_kmh=10),
        dict(temperature_c=25, relative_humidity_pct=math.nan, wind_speed_kmh=10),
        dict(temperature_c=25, relative_humidity_pct=30, wind_speed_kmh=math.nan),
        dict(temperature_c=math.inf, relative_humidity_pct=30, wind_speed_kmh=10),
        dict(temperature_c=-math.inf, relative_humidity_pct=30, wind_speed_kmh=10),
        dict(temperature_c=25, relative_humidity_pct=math.inf, wind_speed_kmh=10),
        dict(temperature_c=25, relative_humidity_pct=30, wind_speed_kmh=-math.inf),
    ],
)
def test_invalid_inputs_are_rejected(input_data):
    with pytest.raises(ValueError):
        FireDangerInput(**input_data)


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (14.999, FireDangerLevel.LOW),
        (15.000, FireDangerLevel.MODERATE),
        (15.001, FireDangerLevel.MODERATE),
        (24.999, FireDangerLevel.MODERATE),
        (25.000, FireDangerLevel.HIGH),
        (25.001, FireDangerLevel.HIGH),
        (39.999, FireDangerLevel.HIGH),
        (40.000, FireDangerLevel.VERY_HIGH),
        (40.001, FireDangerLevel.VERY_HIGH),
        (59.999, FireDangerLevel.VERY_HIGH),
        (60.000, FireDangerLevel.EXTREME),
        (60.001, FireDangerLevel.EXTREME),
        (100.000, FireDangerLevel.EXTREME),
    ],
)
def test_classification_boundaries_use_raw_score(score, expected_level):
    assert _classify_score(score) is expected_level


def test_extreme_mathematical_input_is_clamped_to_max_public_score():
    result = calculate(temperature_c=60, relative_humidity_pct=0, wind_speed_kmh=5000)

    assert result.score == FFWI_MAX_SCORE
    assert result.level is FireDangerLevel.EXTREME
