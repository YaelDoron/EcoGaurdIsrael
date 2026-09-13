"""Fosberg Fire Weather Index score bounds and EcoGuard classifications.

FFWI is a continuous fire-weather index describing how conducive current
weather conditions are to wildfire danger. The threshold constants in this
module are EcoGuard application-level classifications for that continuous
score.

The selected breakpoints correspond to an existing operational FFWI
classification:

- <15 Low
- 15-25 Moderate
- 25-40 High
- 40-60 Very High
- 60-100 Extreme

These thresholds are not claimed to be official Israeli Fire and Rescue
Authority thresholds. Future versions may calibrate classifications using
historical Israeli fire-weather percentiles. Changing classification should
require updating this config, not rewriting the FFWI mathematical calculation.
"""
from __future__ import annotations

FFWI_MIN_SCORE = 0.0
FFWI_MAX_SCORE = 100.0

FFWI_METHODOLOGY_NAME = "FOSBERG_FFWI"
FFWI_METHODOLOGY_VERSION = "1.0"

MODERATE_THRESHOLD = 15.0
HIGH_THRESHOLD = 25.0
VERY_HIGH_THRESHOLD = 40.0
EXTREME_THRESHOLD = 60.0
