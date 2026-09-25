"""Machine-readable reasons for an insufficient_data wildfire-spread result."""
from __future__ import annotations

from enum import Enum


class FireSpreadInsufficientDataReason(Enum):
    """Why FireSpreadInputService could not prepare a READY spread input.

    Set only for INSUFFICIENT_DATA; always None for READY/VALID and
    INACTIVE_EVENT. Persisted rows that predate this field have no reason
    (None = unknown). See backend/docs/fire_spread_prediction.md §9.
    """

    EVENT_UNAVAILABLE = "event_unavailable"
    MISSING_SEVERITY = "missing_severity"
    SEVERITY_NOT_VALID = "severity_not_valid"
    MISSING_WEATHER = "missing_weather"
    INCOMPLETE_WEATHER = "incomplete_weather"
    STALE_WEATHER = "stale_weather"
    FUTURE_WEATHER = "future_weather"
    MISSING_VEGETATION = "missing_vegetation"
    UNSUPPORTED_VEGETATION = "unsupported_vegetation"
