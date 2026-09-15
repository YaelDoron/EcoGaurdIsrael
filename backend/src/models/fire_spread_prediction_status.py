"""Status values for deterministic wildfire-spread prediction runs."""
from __future__ import annotations

from enum import Enum


class FireSpreadPredictionStatus(Enum):
    """Business status of a stored wildfire-spread prediction run."""

    VALID = "valid"
    INSUFFICIENT_DATA = "insufficient_data"
    INACTIVE_EVENT = "inactive_event"
