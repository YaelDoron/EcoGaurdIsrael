"""Explicit rule-vs-ML agreement classification for Fire Detection (Task 5)."""
from __future__ import annotations

from enum import Enum


class FireDetectionMLRuleAgreement(Enum):
    """How the deterministic rule decision and the ML assessment relate.

    Classified using FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD (default
    0.50) purely to interpret the ML probability as fire/no-fire for THIS
    comparison - it is not the escalation threshold used by HYBRID mode.
    """

    AGREE_FIRE = "agree_fire"  # rule says fire (SUSPECTED/CONFIRMED), ML >= threshold
    AGREE_NO_FIRE = "agree_no_fire"  # rule says NO_EVENT, ML < threshold
    RULE_STRONGER = "rule_stronger"  # rule says fire, ML < threshold (rule alone drives the outcome)
    ML_STRONGER = "ml_stronger"  # rule says NO_EVENT, ML >= threshold (ML alone would say fire)
    ML_UNAVAILABLE = "ml_unavailable"  # ML assessment could not be produced
