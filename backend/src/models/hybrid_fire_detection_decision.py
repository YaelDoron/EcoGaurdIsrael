"""Combined rule + ML Fire Detection decision (Task 5)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.fire_detection_ai_assessment import FireDetectionAIAssessment
from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_ref import FireEvidenceRef


@dataclass(frozen=True)
class HybridFireDetectionDecision:
    """The decision FireDetectionAgent actually acts on, plus full rule/ML traceability.

    (Legacy modes.) `final_confidence` always equals `rule_decision.confidence` - the ML
    assessment never changes the persisted confidence number, even when
    HYBRID mode escalates NO_EVENT to SUSPECTED (in that case the escalation
    is explained by `ml_assessment`/`agreement`/`decision_mode`, not by a
    fabricated confidence value). See fire_detection_decision_policy.py and
    backend/docs/fire_detection_runtime_ml.md.

    This is a separate wrapper rather than new fields bolted onto
    FireDetectionDecision, which stays exactly as Task 1 defined it.
    """

    decision_mode: FireDetectionDecisionMode
    final_status: FireDetectionStatus
    final_confidence: float
    final_latitude: float | None
    final_longitude: float | None
    final_supporting_evidence: tuple[FireEvidenceRef, ...]
    rule_decision: FireDetectionDecision
    ml_assessment: FireDetectionMLAssessment
    agreement: FireDetectionMLRuleAgreement
    # Task 9B: set only by AI_HYBRID_V5. In that mode `final_status` is the locked AI policy's status,
    # `final_confidence` is the model probability, and the rule decision is diagnostics only.
    ai_assessment: FireDetectionAIAssessment | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision_mode, FireDetectionDecisionMode):
            raise ValueError(f"decision_mode must be a FireDetectionDecisionMode, got {self.decision_mode!r}")
        if not isinstance(self.final_status, FireDetectionStatus):
            raise ValueError(f"final_status must be a FireDetectionStatus, got {self.final_status!r}")
        if not isinstance(self.rule_decision, FireDetectionDecision):
            raise ValueError(f"rule_decision must be a FireDetectionDecision, got {self.rule_decision!r}")
        if not isinstance(self.ml_assessment, FireDetectionMLAssessment):
            raise ValueError(f"ml_assessment must be a FireDetectionMLAssessment, got {self.ml_assessment!r}")
        if not isinstance(self.agreement, FireDetectionMLRuleAgreement):
            raise ValueError(f"agreement must be a FireDetectionMLRuleAgreement, got {self.agreement!r}")

        if isinstance(self.final_confidence, bool) or not isinstance(self.final_confidence, Real):
            raise ValueError(f"final_confidence must be numeric, got {self.final_confidence!r}")
        if not math.isfinite(self.final_confidence) or not 0.0 <= self.final_confidence <= 1.0:
            raise ValueError(f"final_confidence must be finite within [0, 1], got {self.final_confidence!r}")

        object.__setattr__(
            self,
            "final_supporting_evidence",
            tuple(sorted(set(self.final_supporting_evidence), key=lambda ref: (ref.evidence_type.value, ref.evidence_id))),
        )
        for evidence_ref in self.final_supporting_evidence:
            if not isinstance(evidence_ref, FireEvidenceRef):
                raise ValueError(f"final_supporting_evidence must contain FireEvidenceRef items, got {evidence_ref!r}")

        if self.final_status is FireDetectionStatus.NO_EVENT:
            if self.final_latitude is not None:
                self._validate_coordinate("final_latitude", self.final_latitude, -90, 90)
            if self.final_longitude is not None:
                self._validate_coordinate("final_longitude", self.final_longitude, -180, 180)
        else:
            self._validate_coordinate("final_latitude", self.final_latitude, -90, 90)
            self._validate_coordinate("final_longitude", self.final_longitude, -180, 180)
            if not self.final_supporting_evidence:
                raise ValueError(f"{self.final_status.name} hybrid decisions must include supporting evidence.")

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not minimum <= value <= maximum:
            raise ValueError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}")
