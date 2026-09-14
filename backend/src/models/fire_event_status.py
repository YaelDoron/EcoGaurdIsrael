"""Lifecycle states for persisted wildfire events."""
from __future__ import annotations

from enum import Enum


class FireEventStatus(Enum):
    """Persisted wildfire event lifecycle status."""

    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
