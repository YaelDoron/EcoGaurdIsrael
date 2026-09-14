"""EcoGuard configurable parameters for active-wildfire evidence detection.

These constants are EcoGuard application-level detection parameters. Satellite
confidence labels are mapped to EcoGuard evidence weights; these weights are
not claimed to be statistical probabilities published by NASA. Future
calibration may update these constants without rewriting the core logic.
"""
from __future__ import annotations

MAX_EVIDENCE_DISTANCE_KM = 5.0
MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES = 60

SATELLITE_LOW_CONFIDENCE = 0.40
SATELLITE_NOMINAL_CONFIDENCE = 0.60
SATELLITE_HIGH_CONFIDENCE = 0.75

NEWS_CONFIDENCE = 0.50

SUSPECTED_THRESHOLD = 0.50
CONFIRMED_THRESHOLD = 0.80

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

FIRE_DETECTION_METHODOLOGY_NAME = "ECOGUARD_MULTI_SOURCE_DETECTION"
FIRE_DETECTION_METHODOLOGY_VERSION = "1.0"
