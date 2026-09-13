"""Fire-danger calculation utilities."""

from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_danger.ffwi_config import (
    EXTREME_THRESHOLD,
    FFWI_MAX_SCORE,
    FFWI_MIN_SCORE,
    HIGH_THRESHOLD,
    MODERATE_THRESHOLD,
    VERY_HIGH_THRESHOLD,
)

__all__ = [
    "FFWICalculator",
    "FFWI_MIN_SCORE",
    "FFWI_MAX_SCORE",
    "MODERATE_THRESHOLD",
    "HIGH_THRESHOLD",
    "VERY_HIGH_THRESHOLD",
    "EXTREME_THRESHOLD",
]
