"""Validation tests for the Task 5 runtime ML/hybrid-decision domain models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.hybrid_fire_detection_decision import HybridFireDetectionDecision

UPDATED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


# --- FireDetectionMLAssessment ---


def test_available_assessment_requires_probability_in_range():
    with pytest.raises(ValueError):
        FireDetectionMLAssessment(
            available=True, probability=1.5, model_name="m", model_version="3.0", feature_schema_version="v3", failure_reason=None
        )


def test_available_assessment_rejects_failure_reason():
    with pytest.raises(ValueError):
        FireDetectionMLAssessment(
            available=True, probability=0.5, model_name="m", model_version="3.0", feature_schema_version="v3", failure_reason="oops"
        )


def test_unavailable_assessment_rejects_probability():
    with pytest.raises(ValueError):
        FireDetectionMLAssessment(
            available=False, probability=0.5, model_name=None, model_version=None, feature_schema_version=None, failure_reason="x"
        )


def test_unavailable_assessment_allows_none_failure_reason():
    assessment = FireDetectionMLAssessment(
        available=False, probability=None, model_name=None, model_version=None, feature_schema_version=None, failure_reason=None
    )
    assert assessment.failure_reason is None


# --- HybridFireDetectionDecision ---


def _rule_decision(status, confidence=0.8):
    if status is FireDetectionStatus.NO_EVENT:
        return FireDetectionDecision(confidence=confidence, status=status, latitude=None, longitude=None, supporting_evidence=())
    ref = FireEvidenceRef(FireEvidenceType.SATELLITE, 1)
    return FireDetectionDecision(confidence=confidence, status=status, latitude=32.0, longitude=35.0, supporting_evidence=(ref,))


def _ml_available(probability=0.8):
    return FireDetectionMLAssessment(
        available=True, probability=probability, model_name="m", model_version="3.0", feature_schema_version="v3", failure_reason=None
    )


def test_hybrid_decision_confirmed_requires_location():
    with pytest.raises(ValueError):
        HybridFireDetectionDecision(
            decision_mode=FireDetectionDecisionMode.SHADOW,
            final_status=FireDetectionStatus.CONFIRMED,
            final_confidence=0.8,
            final_latitude=None,
            final_longitude=None,
            final_supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, 1),),
            rule_decision=_rule_decision(FireDetectionStatus.CONFIRMED),
            ml_assessment=_ml_available(),
            agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
        )


def test_hybrid_decision_confirmed_requires_supporting_evidence():
    with pytest.raises(ValueError):
        HybridFireDetectionDecision(
            decision_mode=FireDetectionDecisionMode.SHADOW,
            final_status=FireDetectionStatus.CONFIRMED,
            final_confidence=0.8,
            final_latitude=32.0,
            final_longitude=35.0,
            final_supporting_evidence=(),
            rule_decision=_rule_decision(FireDetectionStatus.CONFIRMED),
            ml_assessment=_ml_available(),
            agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
        )


def test_hybrid_decision_no_event_allows_empty_location_and_evidence():
    decision = HybridFireDetectionDecision(
        decision_mode=FireDetectionDecisionMode.SHADOW,
        final_status=FireDetectionStatus.NO_EVENT,
        final_confidence=0.0,
        final_latitude=None,
        final_longitude=None,
        final_supporting_evidence=(),
        rule_decision=_rule_decision(FireDetectionStatus.NO_EVENT, confidence=0.0),
        ml_assessment=_ml_available(0.1),
        agreement=FireDetectionMLRuleAgreement.AGREE_NO_FIRE,
    )
    assert decision.final_status is FireDetectionStatus.NO_EVENT


def test_hybrid_decision_deduplicates_and_sorts_supporting_evidence():
    ref1 = FireEvidenceRef(FireEvidenceType.SATELLITE, 1)
    ref1_dup = FireEvidenceRef(FireEvidenceType.SATELLITE, 1)
    ref2 = FireEvidenceRef(FireEvidenceType.NEWS, 2)
    decision = HybridFireDetectionDecision(
        decision_mode=FireDetectionDecisionMode.SHADOW,
        final_status=FireDetectionStatus.CONFIRMED,
        final_confidence=0.8,
        final_latitude=32.0,
        final_longitude=35.0,
        final_supporting_evidence=(ref2, ref1, ref1_dup),
        rule_decision=_rule_decision(FireDetectionStatus.CONFIRMED),
        ml_assessment=_ml_available(),
        agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
    )
    assert decision.final_supporting_evidence == (ref2, ref1)  # sorted by (evidence_type.value, evidence_id): "news" < "satellite"


def test_hybrid_decision_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        HybridFireDetectionDecision(
            decision_mode=FireDetectionDecisionMode.SHADOW,
            final_status=FireDetectionStatus.NO_EVENT,
            final_confidence=1.5,
            final_latitude=None,
            final_longitude=None,
            final_supporting_evidence=(),
            rule_decision=_rule_decision(FireDetectionStatus.NO_EVENT, confidence=0.0),
            ml_assessment=_ml_available(0.1),
            agreement=FireDetectionMLRuleAgreement.AGREE_NO_FIRE,
        )


# --- FireEventMLAssessment ---


def test_fire_event_ml_assessment_requires_probability_when_available():
    with pytest.raises(ValueError):
        FireEventMLAssessment(
            fire_event_id=1,
            decision_mode=FireDetectionDecisionMode.SHADOW,
            rule_status=FireDetectionStatus.CONFIRMED,
            rule_confidence=0.8,
            ml_available=True,
            ml_probability=None,
            ml_model_name="m",
            ml_model_version="3.0",
            ml_feature_schema_version="v3",
            ml_failure_reason=None,
            agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
            updated_at=UPDATED_AT,
        )


def test_fire_event_ml_assessment_rejects_probability_when_unavailable():
    with pytest.raises(ValueError):
        FireEventMLAssessment(
            fire_event_id=1,
            decision_mode=FireDetectionDecisionMode.SHADOW,
            rule_status=FireDetectionStatus.CONFIRMED,
            rule_confidence=0.8,
            ml_available=False,
            ml_probability=0.5,
            ml_model_name=None,
            ml_model_version=None,
            ml_feature_schema_version=None,
            ml_failure_reason="x",
            agreement=FireDetectionMLRuleAgreement.ML_UNAVAILABLE,
            updated_at=UPDATED_AT,
        )


def test_fire_event_ml_assessment_requires_positive_id():
    with pytest.raises(ValueError):
        FireEventMLAssessment(
            fire_event_id=0,
            decision_mode=FireDetectionDecisionMode.SHADOW,
            rule_status=FireDetectionStatus.CONFIRMED,
            rule_confidence=0.8,
            ml_available=False,
            ml_probability=None,
            ml_model_name=None,
            ml_model_version=None,
            ml_feature_schema_version=None,
            ml_failure_reason=None,
            agreement=FireDetectionMLRuleAgreement.ML_UNAVAILABLE,
            updated_at=UPDATED_AT,
        )


def test_fire_event_ml_assessment_requires_aware_datetime():
    with pytest.raises(ValueError):
        FireEventMLAssessment(
            fire_event_id=1,
            decision_mode=FireDetectionDecisionMode.SHADOW,
            rule_status=FireDetectionStatus.CONFIRMED,
            rule_confidence=0.8,
            ml_available=False,
            ml_probability=None,
            ml_model_name=None,
            ml_model_version=None,
            ml_feature_schema_version=None,
            ml_failure_reason=None,
            agreement=FireDetectionMLRuleAgreement.ML_UNAVAILABLE,
            updated_at=datetime(2026, 9, 14, 12, 0),  # naive
        )


# --- enum sanity ---


def test_decision_mode_values():
    assert {mode.value for mode in FireDetectionDecisionMode} == {"rule_only", "shadow", "hybrid"}


def test_agreement_values():
    assert {agreement.value for agreement in FireDetectionMLRuleAgreement} == {
        "agree_fire",
        "agree_no_fire",
        "rule_stronger",
        "ml_stronger",
        "ml_unavailable",
    }
