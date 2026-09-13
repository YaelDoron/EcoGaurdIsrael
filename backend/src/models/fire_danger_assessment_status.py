"""Business status for a persisted fire-danger assessment."""
from __future__ import annotations

from enum import Enum


class FireDangerAssessmentStatus(Enum):
    """Whether a fire-danger assessment has a valid calculated result."""

    VALID = "valid"
    INSUFFICIENT_DATA = "insufficient_data"
