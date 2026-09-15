"""Pure active wildfire severity calculation.

Fire severity is calculated only after active-fire detection has identified a
fire. This module does not decide whether a fire exists, does not retrieve
inputs, does not persist results, and does not inspect FireEvent state.

The score is an EcoGuard operational severity score. It is deterministic for
identical inputs and is not a scientifically calibrated probability. FRP is the
dominant current-fire intensity factor. Wind and humidity describe current
environmental fire behavior. Vegetation is optional in Severity v1: when it is
missing, available weights are re-normalized instead of treating missing
vegetation as zero fuel. Terrain is intentionally excluded from Severity v1 and
may be used later by fire-spread prediction.
"""
from __future__ import annotations

from src.calculators.fire_severity.fire_severity_config import (
    CRITICAL_THRESHOLD,
    DRYNESS_WEIGHT,
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
    FRP_REFERENCE_MW,
    FRP_WEIGHT,
    HIGH_THRESHOLD,
    MAX_SEVERITY_SCORE,
    MIN_SEVERITY_SCORE,
    MODERATE_THRESHOLD,
    VEGETATION_WEIGHT,
    WIND_REFERENCE_KMH,
    WIND_WEIGHT,
)
from src.models.fire_severity_calculation import FireSeverityCalculation
from src.models.fire_severity_input import FireSeverityInput
from src.models.fire_severity_level import FireSeverityLevel


class FireSeverityCalculator:
    """Calculate deterministic active wildfire severity from normalized input."""

    def calculate(self, input_data: FireSeverityInput) -> FireSeverityCalculation:
        """Return a severity score, level, factors, and methodology metadata."""
        if not isinstance(input_data, FireSeverityInput):
            raise ValueError(f"input_data must be a FireSeverityInput, got {input_data!r}")

        frp_factor = _clamp_factor(input_data.frp_mw / FRP_REFERENCE_MW)
        wind_factor = _clamp_factor(input_data.wind_speed_kmh / WIND_REFERENCE_KMH)
        dryness_factor = _clamp_factor(1 - (input_data.relative_humidity_pct / 100))
        vegetation_factor = input_data.vegetation_fuel_score

        weighted_sum = (
            (FRP_WEIGHT * frp_factor)
            + (WIND_WEIGHT * wind_factor)
            + (DRYNESS_WEIGHT * dryness_factor)
        )
        available_weight = FRP_WEIGHT + WIND_WEIGHT + DRYNESS_WEIGHT

        if vegetation_factor is not None:
            weighted_sum += VEGETATION_WEIGHT * vegetation_factor
            available_weight += VEGETATION_WEIGHT

        score = _clamp_score(MAX_SEVERITY_SCORE * (weighted_sum / available_weight))

        return FireSeverityCalculation(
            score=score,
            level=_classify_score(score),
            frp_factor=frp_factor,
            wind_factor=wind_factor,
            dryness_factor=dryness_factor,
            vegetation_factor=vegetation_factor,
            available_weight=available_weight,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        )


def _clamp_factor(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _clamp_score(score: float) -> float:
    return min(max(score, MIN_SEVERITY_SCORE), MAX_SEVERITY_SCORE)


def _classify_score(score: float) -> FireSeverityLevel:
    if score < MODERATE_THRESHOLD:
        return FireSeverityLevel.LOW
    if score < HIGH_THRESHOLD:
        return FireSeverityLevel.MODERATE
    if score < CRITICAL_THRESHOLD:
        return FireSeverityLevel.HIGH
    return FireSeverityLevel.CRITICAL
