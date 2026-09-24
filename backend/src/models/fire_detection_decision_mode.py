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

    AI_HYBRID_V5 (Task 9B): a different KIND of mode, not a variant of HYBRID. The
    FireEvent status comes ONLY from the approved HistGradientBoosting V5 model
    (V5 history-aware features -> P(fire)) and the locked AI Hybrid Policy v5.0
    (P < 0.40 NO_EVENT; P >= 0.40 SUSPECTED; P >= 0.80 with >= 2 current satellite
    pixels CONFIRMED). The rule calculator may still run for diagnostics but can
    never influence the status, and there is NO silent fallback to rules: an
    unusable artifact fails the detection cycle explicitly. Opt-in only; it is
    not the default. See FireDetectionAIHybridClassifierV5.
    """

    RULE_ONLY = "rule_only"
    SHADOW = "shadow"
    HYBRID = "hybrid"
    AI_HYBRID_V5 = "ai_hybrid_v5"
