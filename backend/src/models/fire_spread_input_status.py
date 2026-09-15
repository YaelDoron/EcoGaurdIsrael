"""Input preparation status for wildfire-spread prediction."""
from __future__ import annotations

from enum import Enum


class FireSpreadInputStatus(Enum):
    """Readiness state for preparing FireSpreadInput."""

    READY = "ready"
    INSUFFICIENT_DATA = "insufficient_data"
    INACTIVE_EVENT = "inactive_event"
