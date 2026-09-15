"""Tests for the pure active wildfire severity calculator."""
from __future__ import annotations

import pytest

from src.calculators.fire_severity.fire_severity_calculator import (
    FireSeverityCalculator,
    _classify_score,
)
from src.calculators.fire_severity.fire_severity_config import (
    CRITICAL_THRESHOLD,
    DRYNESS_WEIGHT,
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
    FRP_REFERENCE_MW,
    FRP_WEIGHT,
    HIGH_THRESHOLD,
    MAX_SEVERITY_SCORE,
    MODERATE_THRESHOLD,
    VEGETATION_WEIGHT,
    WIND_REFERENCE_KMH,
    WIND_WEIGHT,
)
from src.models import FireSeverityInput, FireSeverityLevel


def calculate(
    *,
    frp_mw: float,
    wind_speed_kmh: float,
    relative_humidity_pct: float,
    vegetation_fuel_score: float | None = None,
):
    return FireSeverityCalculator().calculate(
        FireSeverityInput(
            frp_mw=frp_mw,
            wind_speed_kmh=wind_speed_kmh,
            relative_humidity_pct=relative_humidity_pct,
            vegetation_fuel_score=vegetation_fuel_score,
        )
    )


def expected_score(
    *,
    frp_factor: float,
    wind_factor: float,
    dryness_factor: float,
    vegetation_factor: float | None,
) -> float:
    weighted_sum = (
        FRP_WEIGHT * frp_factor
        + WIND_WEIGHT * wind_factor
        + DRYNESS_WEIGHT * dryness_factor
    )
    available_weight = FRP_WEIGHT + WIND_WEIGHT + DRYNESS_WEIGHT
    if vegetation_factor is not None:
        weighted_sum += VEGETATION_WEIGHT * vegetation_factor
        available_weight += VEGETATION_WEIGHT
    return MAX_SEVERITY_SCORE * (weighted_sum / available_weight)


def test_all_zero_threat_factors_are_low():
    result = calculate(frp_mw=0, wind_speed_kmh=0, relative_humidity_pct=100)

    assert result.score == pytest.approx(0)
    assert result.level is FireSeverityLevel.LOW
    assert result.frp_factor == pytest.approx(0)
    assert result.wind_factor == pytest.approx(0)
    assert result.dryness_factor == pytest.approx(0)


def test_maximum_normalized_factors_are_critical():
    result = calculate(
        frp_mw=FRP_REFERENCE_MW,
        wind_speed_kmh=WIND_REFERENCE_KMH,
        relative_humidity_pct=0,
        vegetation_fuel_score=1,
    )

    assert result.score == pytest.approx(100)
    assert result.level is FireSeverityLevel.CRITICAL


def test_factors_above_normalization_reference_are_clamped_and_score_stays_bounded():
    result = calculate(
        frp_mw=FRP_REFERENCE_MW * 2,
        wind_speed_kmh=WIND_REFERENCE_KMH * 3,
        relative_humidity_pct=0,
        vegetation_fuel_score=1,
    )

    assert result.frp_factor == pytest.approx(1)
    assert result.wind_factor == pytest.approx(1)
    assert result.score == pytest.approx(100)
    assert result.score <= MAX_SEVERITY_SCORE


def test_strong_wind_low_humidity_high_frp_scores_higher_than_mild_conditions():
    severe = calculate(
        frp_mw=90,
        wind_speed_kmh=45,
        relative_humidity_pct=10,
        vegetation_fuel_score=0.6,
    )
    mild = calculate(
        frp_mw=20,
        wind_speed_kmh=10,
        relative_humidity_pct=70,
        vegetation_fuel_score=0.6,
    )

    assert severe.score > mild.score


def test_vegetation_one_contributes_according_to_optional_weight():
    result = calculate(
        frp_mw=50,
        wind_speed_kmh=25,
        relative_humidity_pct=50,
        vegetation_fuel_score=1,
    )

    assert result.available_weight == pytest.approx(1.0)
    assert result.vegetation_factor == pytest.approx(1)
    assert result.score == pytest.approx(
        expected_score(
            frp_factor=0.5,
            wind_factor=0.5,
            dryness_factor=0.5,
            vegetation_factor=1,
        )
    )


def test_vegetation_zero_is_valid_and_different_from_missing_vegetation():
    missing = calculate(
        frp_mw=50,
        wind_speed_kmh=25,
        relative_humidity_pct=50,
        vegetation_fuel_score=None,
    )
    zero = calculate(
        frp_mw=50,
        wind_speed_kmh=25,
        relative_humidity_pct=50,
        vegetation_fuel_score=0,
    )

    assert missing.available_weight == pytest.approx(0.90)
    assert zero.available_weight == pytest.approx(1.00)
    assert missing.score != zero.score
    assert missing.score == pytest.approx(
        expected_score(
            frp_factor=0.5,
            wind_factor=0.5,
            dryness_factor=0.5,
            vegetation_factor=None,
        )
    )
    assert zero.score == pytest.approx(
        expected_score(
            frp_factor=0.5,
            wind_factor=0.5,
            dryness_factor=0.5,
            vegetation_factor=0,
        )
    )


def test_missing_vegetation_does_not_crash_and_is_renormalized_not_treated_as_zero():
    result = calculate(
        frp_mw=80,
        wind_speed_kmh=40,
        relative_humidity_pct=20,
    )

    assert result.vegetation_factor is None
    assert result.available_weight == pytest.approx(0.90)
    assert result.score == pytest.approx(
        expected_score(
            frp_factor=0.8,
            wind_factor=0.8,
            dryness_factor=0.8,
            vegetation_factor=None,
        )
    )


@pytest.mark.parametrize(
    ("frp_mw", "expected_factor"),
    [
        (0, 0),
        (FRP_REFERENCE_MW / 2, 0.5),
        (FRP_REFERENCE_MW, 1),
        (FRP_REFERENCE_MW + 25, 1),
    ],
)
def test_frp_factor_normalization(frp_mw, expected_factor):
    result = calculate(frp_mw=frp_mw, wind_speed_kmh=0, relative_humidity_pct=100)

    assert result.frp_factor == pytest.approx(expected_factor)


@pytest.mark.parametrize(
    ("wind_speed_kmh", "expected_factor"),
    [
        (0, 0),
        (WIND_REFERENCE_KMH / 2, 0.5),
        (WIND_REFERENCE_KMH, 1),
        (WIND_REFERENCE_KMH + 10, 1),
    ],
)
def test_wind_factor_normalization(wind_speed_kmh, expected_factor):
    result = calculate(frp_mw=0, wind_speed_kmh=wind_speed_kmh, relative_humidity_pct=100)

    assert result.wind_factor == pytest.approx(expected_factor)


@pytest.mark.parametrize(
    ("relative_humidity_pct", "expected_dryness"),
    [
        (100, 0),
        (50, 0.5),
        (0, 1),
    ],
)
def test_humidity_to_dryness_factor(relative_humidity_pct, expected_dryness):
    result = calculate(
        frp_mw=0,
        wind_speed_kmh=0,
        relative_humidity_pct=relative_humidity_pct,
    )

    assert result.dryness_factor == pytest.approx(expected_dryness)


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (MODERATE_THRESHOLD - 0.001, FireSeverityLevel.LOW),
        (MODERATE_THRESHOLD, FireSeverityLevel.MODERATE),
        (HIGH_THRESHOLD - 0.001, FireSeverityLevel.MODERATE),
        (HIGH_THRESHOLD, FireSeverityLevel.HIGH),
        (CRITICAL_THRESHOLD - 0.001, FireSeverityLevel.HIGH),
        (CRITICAL_THRESHOLD, FireSeverityLevel.CRITICAL),
        (100, FireSeverityLevel.CRITICAL),
    ],
)
def test_classification_boundaries(score, expected_level):
    assert _classify_score(score) is expected_level


def test_identical_inputs_produce_identical_outputs_repeatedly():
    input_data = FireSeverityInput(
        frp_mw=42,
        wind_speed_kmh=18,
        relative_humidity_pct=35,
        vegetation_fuel_score=0.4,
    )
    calculator = FireSeverityCalculator()

    first = calculator.calculate(input_data)
    second = calculator.calculate(input_data)

    assert first == second


def test_result_includes_methodology_metadata_from_config():
    result = calculate(
        frp_mw=10,
        wind_speed_kmh=10,
        relative_humidity_pct=50,
    )

    assert result.methodology == FIRE_SEVERITY_METHODOLOGY_NAME
    assert result.methodology_version == FIRE_SEVERITY_METHODOLOGY_VERSION


def test_calculator_rejects_non_severity_input():
    with pytest.raises(ValueError):
        FireSeverityCalculator().calculate("not input")
