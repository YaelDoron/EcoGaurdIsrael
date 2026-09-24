"""Combine a rule-based FireDetectionDecision with a FireDetectionMLAssessment (Task 5).

Pure decision logic - no I/O, no sklearn, no repositories. Given already-
computed rule/ML results (and, only for the rare HYBRID escalation path, the
candidate's raw evidence to derive a location), returns one
HybridFireDetectionDecision that FireDetectionAgent acts on directly.

Safety invariants enforced here, always, regardless of configuration:
  - RULE_ONLY/SHADOW: final status/confidence/location ALWAYS equal the rule
    decision's. ML can never change what gets persisted in these modes.
  - HYBRID: a SUSPECTED/CONFIRMED rule decision is NEVER changed by ML - not
    downgraded, and SUSPECTED is never promoted to CONFIRMED by ML alone.
    Only a NO_EVENT rule decision may be promoted, and only to SUSPECTED
    (never directly to CONFIRMED), and only when ML is available and at or
    above the configured suspect threshold.
  - `final_confidence` always equals `rule_decision.confidence` in every
    mode - ML never fabricates a confidence value.
"""
from __future__ import annotations

from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.hybrid_fire_detection_decision import HybridFireDetectionDecision

DEFAULT_ML_CLASSIFICATION_THRESHOLD = 0.50


def estimate_candidate_location(evidence_items: tuple[FireDetectionEvidence, ...]) -> tuple[float, float]:
    """Same satellite-priority centroid FireDetectionCalculator uses for SUSPECTED/CONFIRMED.

    Public so the AI Hybrid V5 path (which must locate a candidate even when the rule calculator says NO_EVENT) matches
    events with exactly the same coordinates the rule path would use.
    """
    satellite_items = tuple(item for item in evidence_items if item.evidence_type is FireEvidenceType.SATELLITE)
    items = satellite_items or tuple(item for item in evidence_items if item.evidence_type is FireEvidenceType.NEWS)
    return (
        sum(item.latitude for item in items) / len(items),
        sum(item.longitude for item in items) / len(items),
    )


class FireDetectionHybridPolicy:
    """Decide the final Fire Detection outcome for one runtime decision_mode.

    ml_suspect_threshold=None means HYBRID NO_EVENT->SUSPECTED escalation is
    disabled (e.g. no threshold met the precision target in
    scripts.analyze_fire_detection_ml_threshold) - HYBRID then behaves
    exactly like SHADOW for NO_EVENT rule decisions.
    """

    def __init__(
        self,
        mode: FireDetectionDecisionMode,
        ml_classification_threshold: float = DEFAULT_ML_CLASSIFICATION_THRESHOLD,
        ml_suspect_threshold: float | None = None,
    ) -> None:
        if not isinstance(mode, FireDetectionDecisionMode):
            raise ValueError(f"mode must be a FireDetectionDecisionMode, got {mode!r}")
        if mode is FireDetectionDecisionMode.AI_HYBRID_V5:
            # This policy only knows the rule-driven modes; silently treating AI_HYBRID_V5 like SHADOW would label
            # rule decisions as AI decisions. The V5 path lives in FireDetectionAIHybridClassifierV5.
            raise ValueError(
                "FireDetectionHybridPolicy does not implement ai_hybrid_v5; "
                "the AI Hybrid V5 status comes from FireDetectionAIHybridClassifierV5."
            )
        self._mode = mode
        self._ml_classification_threshold = ml_classification_threshold
        self._ml_suspect_threshold = ml_suspect_threshold

    @property
    def mode(self) -> FireDetectionDecisionMode:
        return self._mode

    def decide(
        self,
        rule_decision: FireDetectionDecision,
        ml_assessment: FireDetectionMLAssessment,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> HybridFireDetectionDecision:
        agreement = self._classify_agreement(rule_decision, ml_assessment)

        if self._mode is FireDetectionDecisionMode.HYBRID:
            final_status, final_latitude, final_longitude = self._hybrid_status(
                rule_decision, ml_assessment, candidate_evidence
            )
        else:
            final_status = rule_decision.status
            final_latitude = rule_decision.latitude
            final_longitude = rule_decision.longitude

        escalated = final_status is not rule_decision.status
        final_supporting_evidence = (
            self._candidate_evidence_refs(candidate_evidence) if escalated else rule_decision.supporting_evidence
        )

        return HybridFireDetectionDecision(
            decision_mode=self._mode,
            final_status=final_status,
            final_confidence=rule_decision.confidence,
            final_latitude=final_latitude,
            final_longitude=final_longitude,
            final_supporting_evidence=final_supporting_evidence,
            rule_decision=rule_decision,
            ml_assessment=ml_assessment,
            agreement=agreement,
        )

    @staticmethod
    def _candidate_evidence_refs(candidate_evidence: tuple[FireDetectionEvidence, ...]) -> tuple[FireEvidenceRef, ...]:
        return tuple(
            FireEvidenceRef(evidence_type=item.evidence_type, evidence_id=item.evidence_id)
            for item in candidate_evidence
        )

    def _classify_agreement(
        self,
        rule_decision: FireDetectionDecision,
        ml_assessment: FireDetectionMLAssessment,
    ) -> FireDetectionMLRuleAgreement:
        if not ml_assessment.available:
            return FireDetectionMLRuleAgreement.ML_UNAVAILABLE

        rule_says_fire = rule_decision.status is not FireDetectionStatus.NO_EVENT
        ml_says_fire = ml_assessment.probability >= self._ml_classification_threshold

        if rule_says_fire and ml_says_fire:
            return FireDetectionMLRuleAgreement.AGREE_FIRE
        if not rule_says_fire and not ml_says_fire:
            return FireDetectionMLRuleAgreement.AGREE_NO_FIRE
        if rule_says_fire and not ml_says_fire:
            return FireDetectionMLRuleAgreement.RULE_STRONGER
        return FireDetectionMLRuleAgreement.ML_STRONGER

    def _hybrid_status(
        self,
        rule_decision: FireDetectionDecision,
        ml_assessment: FireDetectionMLAssessment,
        candidate_evidence: tuple[FireDetectionEvidence, ...],
    ) -> tuple[FireDetectionStatus, float | None, float | None]:
        # SUSPECTED/CONFIRMED rule decisions are never changed by ML: not
        # downgraded, and SUSPECTED is never auto-promoted to CONFIRMED by
        # ML alone.
        if rule_decision.status is not FireDetectionStatus.NO_EVENT:
            return rule_decision.status, rule_decision.latitude, rule_decision.longitude

        if (
            self._ml_suspect_threshold is None
            or not ml_assessment.available
            or ml_assessment.probability < self._ml_suspect_threshold
        ):
            return FireDetectionStatus.NO_EVENT, None, None

        latitude, longitude = self._estimate_escalation_location(candidate_evidence)
        return FireDetectionStatus.SUSPECTED, latitude, longitude

    @staticmethod
    def _estimate_escalation_location(
        evidence_items: tuple[FireDetectionEvidence, ...],
    ) -> tuple[float, float]:
        return estimate_candidate_location(evidence_items)
