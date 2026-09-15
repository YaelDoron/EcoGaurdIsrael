"""EcoGuard configurable parameters for active wildfire severity.

These constants are EcoGuard application-level methodology parameters. They
are configurable and are not official emergency-authority thresholds.

`FRP_REFERENCE_MW` and `WIND_REFERENCE_KMH` are normalization reference values.
Future empirical calibration may change these values without changing the
calculator's pure responsibility boundary.
"""
from __future__ import annotations

FRP_WEIGHT = 0.50
WIND_WEIGHT = 0.20
DRYNESS_WEIGHT = 0.20
VEGETATION_WEIGHT = 0.10

FRP_REFERENCE_MW = 100.0
WIND_REFERENCE_KMH = 50.0

MIN_SEVERITY_SCORE = 0.0
MAX_SEVERITY_SCORE = 100.0

LOW_MIN = 0.0
MODERATE_THRESHOLD = 25.0
HIGH_THRESHOLD = 50.0
CRITICAL_THRESHOLD = 75.0
MAX_LEVEL_SCORE = 100.0

FIRE_SEVERITY_METHODOLOGY_NAME = "ECOGUARD_ACTIVE_FIRE_SEVERITY"
FIRE_SEVERITY_METHODOLOGY_VERSION = "1.0"
