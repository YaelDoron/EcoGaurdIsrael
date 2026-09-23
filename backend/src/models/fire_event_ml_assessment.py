"""Domain model for one FireEvent's latest runtime ML/decision trace (Task 5)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus


@dataclass(frozen=True)
class FireEventMLAssessment:
    """Latest runtime ML/decision trace for one FireEvent (observability only).

    One row per FireEvent, upserted on each reevaluation - NOT a history
    log. `rule_status`/`rule_confidence` are a snapshot of the deterministic
    rule decision at the last evaluation: FireEvent.status/detection_confidence
    already hold the current values used for RULE_ONLY/SHADOW, but in HYBRID
    mode FireEvent.status can diverge from the rule status (ML-driven
    NO_EVENT -> SUSPECTED escalation) - this snapshot keeps "what did the
    rule say" answerable without duplicating FireEvent's fields for their
    own sake in the common case.
    """

    fire_event_id: int
    decision_mode: FireDetectionDecisionMode
    rule_status: FireDetectionStatus
    rule_confidence: float
    ml_available: bool
    ml_probability: float | None
    ml_model_name: str | None
    ml_model_version: str | None
    ml_feature_schema_version: str | None
    ml_failure_reason: str | None
    agreement: FireDetectionMLRuleAgreement
    updated_at: datetime

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if not isinstance(self.decision_mode, FireDetectionDecisionMode):
            raise ValueError(f"decision_mode must be a FireDetectionDecisionMode, got {self.decision_mode!r}")
        if not isinstance(self.rule_status, FireDetectionStatus):
            raise ValueError(f"rule_status must be a FireDetectionStatus, got {self.rule_status!r}")
        if isinstance(self.rule_confidence, bool) or not isinstance(self.rule_confidence, Real):
            raise ValueError(f"rule_confidence must be numeric, got {self.rule_confidence!r}")
        if not math.isfinite(self.rule_confidence) or not 0.0 <= self.rule_confidence <= 1.0:
            raise ValueError(f"rule_confidence must be finite within [0, 1], got {self.rule_confidence!r}")
        if not isinstance(self.agreement, FireDetectionMLRuleAgreement):
            raise ValueError(f"agreement must be a FireDetectionMLRuleAgreement, got {self.agreement!r}")
        if not isinstance(self.updated_at, datetime) or self.updated_at.tzinfo is None:
            raise ValueError(f"updated_at must be a timezone-aware datetime, got {self.updated_at!r}")

        if self.ml_available:
            if (
                self.ml_probability is None
                or isinstance(self.ml_probability, bool)
                or not isinstance(self.ml_probability, Real)
                or not math.isfinite(self.ml_probability)
                or not 0.0 <= self.ml_probability <= 1.0
            ):
                raise ValueError(f"ml_probability must be within [0, 1] when ml_available, got {self.ml_probability!r}")
        elif self.ml_probability is not None:
            raise ValueError("ml_probability must be None when not ml_available.")
