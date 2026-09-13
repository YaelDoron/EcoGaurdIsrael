"""Status of fire-danger input preparation."""
from __future__ import annotations

from enum import Enum


class FireDangerInputStatus(Enum):
    """Whether enough weather data exists to calculate fire danger."""

    READY = "ready"
    INSUFFICIENT_DATA = "insufficient_data"
