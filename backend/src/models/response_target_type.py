"""Operational response target categories."""
from __future__ import annotations

from enum import Enum


class ResponseTargetType(Enum):
    """EcoGuard response target type for downstream operations planning."""

    ACTIVE_FIRE = "active_fire"
    PREDICTED_RISK = "predicted_risk"
