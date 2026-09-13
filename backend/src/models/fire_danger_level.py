"""Application-level fire-danger classifications."""
from __future__ import annotations

from enum import Enum


class FireDangerLevel(Enum):
    """EcoGuard classification for a fire-weather danger score."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"
