"""Application-level active wildfire severity classifications."""
from __future__ import annotations

from enum import Enum


class FireSeverityLevel(Enum):
    """EcoGuard classification for an active wildfire severity score."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
