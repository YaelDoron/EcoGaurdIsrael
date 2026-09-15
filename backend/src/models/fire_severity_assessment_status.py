"""Persistence status for active wildfire severity assessments."""
from __future__ import annotations

from enum import Enum


class FireSeverityAssessmentStatus(Enum):
    """Business status of a stored severity assessment."""

    VALID = "valid"
    INSUFFICIENT_DATA = "insufficient_data"
    INACTIVE_EVENT = "inactive_event"
