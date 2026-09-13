"""Configuration for fire-danger input selection.

These constants are EcoGuard input-selection rules. They are not part of the
Fosberg Fire Weather Index scientific formula, and they may be tuned later
without changing FFWICalculator.
"""
from __future__ import annotations

MAX_WEATHER_AGE_MINUTES = 30
MIN_VALID_STATIONS = 1
