"""Pure Fosberg Fire Weather Index calculation.

FFWI evaluates how conducive current weather conditions are to wildfire
danger. It does not determine whether there is currently an active wildfire.
Satellite hotspots and news reports are not FFWI inputs; they belong to future
fire-detection analysis. Vegetation/fuel context is also outside the base
Fosberg calculation and may be combined at a higher application layer later.

This module does not implement an official Israeli fire-danger rating.
"""
from __future__ import annotations

import math

from src.calculators.fire_danger.ffwi_config import (
    EXTREME_THRESHOLD,
    FFWI_MAX_SCORE,
    FFWI_MIN_SCORE,
    HIGH_THRESHOLD,
    MODERATE_THRESHOLD,
    VERY_HIGH_THRESHOLD,
)
from src.models.fire_danger_calculation import FireDangerCalculation
from src.models.fire_danger_input import FireDangerInput
from src.models.fire_danger_level import FireDangerLevel

_KMH_PER_MPH = 1.609344


class FFWICalculator:
    """Calculate Fosberg Fire Weather Index from normalized weather inputs."""

    def calculate(self, input_data: FireDangerInput) -> FireDangerCalculation:
        """Return a deterministic FFWI score and EcoGuard classification."""
        if not isinstance(input_data, FireDangerInput):
            raise ValueError(f"input_data must be a FireDangerInput, got {input_data!r}")

        temperature_f = _celsius_to_fahrenheit(input_data.temperature_c)
        wind_speed_mph = _kmh_to_mph(input_data.wind_speed_kmh)
        moisture = _equilibrium_moisture_content(
            relative_humidity_pct=input_data.relative_humidity_pct,
            temperature_f=temperature_f,
        )
        eta = _moisture_damping_coefficient(moisture)
        raw_score = eta * math.sqrt(1 + wind_speed_mph**2) / 0.3002
        score = _clamp_score(raw_score)
        return FireDangerCalculation(score=score, level=_classify_score(score))


def _celsius_to_fahrenheit(temperature_c: float) -> float:
    return temperature_c * 9 / 5 + 32


def _kmh_to_mph(wind_speed_kmh: float) -> float:
    return wind_speed_kmh / _KMH_PER_MPH


def _equilibrium_moisture_content(relative_humidity_pct: float, temperature_f: float) -> float:
    h = relative_humidity_pct
    if h < 10:
        return 0.03229 + 0.281073 * h - 0.000578 * h * temperature_f
    if h <= 50:
        return 2.22749 + 0.160107 * h - 0.01478 * temperature_f
    return 21.0606 + 0.005565 * h**2 - 0.00035 * h * temperature_f - 0.483199 * h


def _moisture_damping_coefficient(moisture_content: float) -> float:
    x = moisture_content / 30
    return 1 - 2 * x + 1.5 * x**2 - 0.5 * x**3


def _clamp_score(score: float) -> float:
    return min(max(score, FFWI_MIN_SCORE), FFWI_MAX_SCORE)


def _classify_score(score: float) -> FireDangerLevel:
    if score < MODERATE_THRESHOLD:
        return FireDangerLevel.LOW
    if score < HIGH_THRESHOLD:
        return FireDangerLevel.MODERATE
    if score < VERY_HIGH_THRESHOLD:
        return FireDangerLevel.HIGH
    if score < EXTREME_THRESHOLD:
        return FireDangerLevel.VERY_HIGH
    return FireDangerLevel.EXTREME
