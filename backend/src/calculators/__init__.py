"""Pure deterministic business and mathematical calculators."""

from src.calculators.fire_detection import FireDetectionCalculator
from src.calculators.fire_danger import FFWICalculator

__all__ = ["FFWICalculator", "FireDetectionCalculator"]
