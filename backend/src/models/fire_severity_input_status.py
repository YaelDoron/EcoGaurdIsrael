"""Input preparation status for active wildfire severity."""
from __future__ import annotations

from enum import Enum


class FireSeverityInputStatus(Enum):
    """Readiness state for preparing FireSeverityInput."""

    READY = "ready"
    INSUFFICIENT_DATA = "insufficient_data"
    INACTIVE_EVENT = "inactive_event"
