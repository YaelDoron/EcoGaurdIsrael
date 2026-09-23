"""Per-candidate rule/ML/decision trace for one FireDetectionAgent.detect() run (Task 5)."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus


@dataclass(frozen=True)
class FireDetectionCandidateAssessment:
    """Observability record for one evaluated candidate - not persisted as-is.

    Exists so callers (the demo simulation output, future API/UI) can show
    that ML inference really ran, without needing FireDetectionAgent to
    return raw HybridFireDetectionDecision/sklearn-adjacent objects.
    `event_id` is None for a NO_EVENT final decision (no FireEvent involved).
    """

    event_id: int | None
    rule_status: FireDetectionStatus
    rule_confidence: float
    final_status: FireDetectionStatus
    decision_mode: FireDetectionDecisionMode
    ml_available: bool
    ml_probability: float | None
    ml_model_version: str | None
    agreement: FireDetectionMLRuleAgreement
