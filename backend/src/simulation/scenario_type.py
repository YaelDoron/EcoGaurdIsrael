"""Simulation scenario categories."""
from __future__ import annotations

from enum import Enum


class ScenarioType(Enum):
    """High-level outcome/context for a simulated demo scenario."""

    LOW_RISK_NO_FIRE = "low_risk_no_fire"
    MODERATE_RISK_NO_FIRE = "moderate_risk_no_fire"
    HIGH_RISK_NO_FIRE = "high_risk_no_fire"
    ACTIVE_FIRE = "active_fire"
