"""Runtime Fire Detection decision mode (Task 5 hybrid ML integration)."""
from __future__ import annotations

from enum import Enum


class FireDetectionDecisionMode(Enum):
    """How the ML assessment may influence the final Fire Detection decision.

    RULE_ONLY: historical behavior - the deterministic FireDetectionCalculator
    only. ML is not loaded or called.

    SHADOW (default): both the rule-based calculator and the ML model run,
    but the final status always equals the rule decision's status. The ML
    assessment is recorded for observability/analysis only - it can never
    change what FireEvent gets created/updated.

    HYBRID: both run, and a conservative policy may let ML promote a
    NO_EVENT rule decision to SUSPECTED (never CONFIRMED, never downgrade a
    SUSPECTED/CONFIRMED rule decision). See FireDetectionHybridPolicy.
    """

    RULE_ONLY = "rule_only"
    SHADOW = "shadow"
    HYBRID = "hybrid"
