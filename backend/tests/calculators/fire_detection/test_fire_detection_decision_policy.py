"""Tests for FireDetectionHybridPolicy - decision-mode and agreement behavior."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType

OBSERVED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
LATITUDE = 32.731
LONGITUDE = 35.046
SUSPECT_THRESHOLD = 0.70


def satellite_evidence(evidence_id=1) -> tuple[FireDetectionEvidence, ...]:
    return (
        FireDetectionEvidence(
            evidence_id=evidence_id,
            evidence_type=FireEvidenceType.SATELLITE,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            observed_at=OBSERVED_AT,
            satellite_confidence="nominal",
        ),
    )


def rule_decision(status, confidence, with_location=True, evidence_id=1) -> FireDetectionDecision:
    if status is FireDetectionStatus.NO_EVENT:
        return FireDetectionDecision(
            confidence=confidence, status=status, latitude=None, longitude=None, supporting_evidence=()
        )
    ref = FireEvidenceRef(FireEvidenceType.SATELLITE, evidence_id)
    return FireDetectionDecision(
        confidence=confidence,
        status=status,
        latitude=LATITUDE if with_location else None,
        longitude=LONGITUDE if with_location else None,
        supporting_evidence=(ref,),
    )


def ml_result(probability, available=True, failure_reason=None) -> FireDetectionMLAssessment:
    if not available:
        return FireDetectionMLAssessment(
            available=False,
            probability=None,
            model_name=None,
            model_version=None,
            feature_schema_version=None,
            failure_reason=failure_reason or "unavailable",
        )
    return FireDetectionMLAssessment(
        available=True,
        probability=probability,
        model_name="fire_detection_logistic_v3",
        model_version="3.0",
        feature_schema_version="v3",
        failure_reason=None,
    )


def make_policy(mode, suspect_threshold=SUSPECT_THRESHOLD, classification_threshold=0.50) -> FireDetectionHybridPolicy:
    return FireDetectionHybridPolicy(mode=mode, ml_classification_threshold=classification_threshold, ml_suspect_threshold=suspect_threshold)


# --- RULE_ONLY ---


def test_rule_only_final_equals_rule_regardless_of_ml():
    policy = make_policy(FireDetectionDecisionMode.RULE_ONLY)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)

    decision = policy.decide(rule, ml_result(0.05), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.CONFIRMED
    assert decision.final_confidence == pytest.approx(0.8)


def test_rule_only_no_event_stays_no_event_even_with_high_ml():
    policy = make_policy(FireDetectionDecisionMode.RULE_ONLY)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.99), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.NO_EVENT


# --- SHADOW ---


def test_shadow_final_always_equals_rule_high_ml_low_rule():
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.95), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.NO_EVENT
    assert decision.decision_mode is FireDetectionDecisionMode.SHADOW


def test_shadow_final_always_equals_rule_low_ml_high_rule():
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)

    decision = policy.decide(rule, ml_result(0.05), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.CONFIRMED


def test_shadow_preserves_ml_assessment_metadata():
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)
    ml = ml_result(0.86)

    decision = policy.decide(rule, ml, satellite_evidence())

    assert decision.ml_assessment == ml
    assert decision.ml_assessment.probability == pytest.approx(0.86)


# --- HYBRID: CONFIRMED rule ---


def test_hybrid_confirmed_stays_confirmed_regardless_of_ml():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)

    decision = policy.decide(rule, ml_result(0.01), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.CONFIRMED


# --- HYBRID: SUSPECTED rule ---


def test_hybrid_suspected_never_downgraded():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.SUSPECTED, 0.6)

    decision = policy.decide(rule, ml_result(0.02), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.SUSPECTED


def test_hybrid_suspected_never_auto_confirmed_by_ml_alone():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.SUSPECTED, 0.6)

    decision = policy.decide(rule, ml_result(0.99), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.SUSPECTED  # not CONFIRMED


# --- HYBRID: NO_EVENT rule ---


def test_hybrid_no_event_stays_no_event_below_suspect_threshold():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.5), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.NO_EVENT


def test_hybrid_no_event_escalates_to_suspected_above_threshold():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.91), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.SUSPECTED
    assert decision.final_latitude is not None
    assert decision.final_longitude is not None
    assert decision.final_supporting_evidence  # escalation must carry the candidate's evidence


def test_hybrid_no_event_escalation_never_reaches_confirmed():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(1.0), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.SUSPECTED  # never CONFIRMED


def test_hybrid_no_event_escalation_disabled_when_threshold_is_none():
    policy = make_policy(FireDetectionDecisionMode.HYBRID, suspect_threshold=None)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.99), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.NO_EVENT


# --- HYBRID: ML unavailable ---


def test_hybrid_ml_unavailable_final_equals_rule():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)

    decision = policy.decide(rule, ml_result(None, available=False), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.CONFIRMED
    assert decision.agreement is FireDetectionMLRuleAgreement.ML_UNAVAILABLE


def test_hybrid_ml_unavailable_no_event_stays_no_event():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(None, available=False), satellite_evidence())

    assert decision.final_status is FireDetectionStatus.NO_EVENT


# --- confidence never fabricated by ML ---


def test_final_confidence_always_equals_rule_confidence_even_when_escalated():
    policy = make_policy(FireDetectionDecisionMode.HYBRID)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.95), satellite_evidence())

    assert decision.final_confidence == pytest.approx(0.0)  # unchanged rule confidence, not ML probability


# --- agreement classification (Part 40 disagreement coverage) ---


def test_agreement_agree_fire():
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)

    decision = policy.decide(rule, ml_result(0.9), satellite_evidence())

    assert decision.agreement is FireDetectionMLRuleAgreement.AGREE_FIRE


def test_agreement_agree_no_fire():
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.1), satellite_evidence())

    assert decision.agreement is FireDetectionMLRuleAgreement.AGREE_NO_FIRE


def test_agreement_rule_stronger_confirmed_vs_low_ml():
    """Rule CONFIRMED, ML predicts non-fire."""
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.CONFIRMED, 0.8)

    decision = policy.decide(rule, ml_result(0.05), satellite_evidence())

    assert decision.agreement is FireDetectionMLRuleAgreement.RULE_STRONGER
    assert decision.final_status is FireDetectionStatus.CONFIRMED  # disagreement does not raise or change outcome


def test_agreement_ml_stronger_no_event_vs_high_ml():
    """Rule NO_EVENT, ML predicts fire."""
    policy = make_policy(FireDetectionDecisionMode.SHADOW)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy.decide(rule, ml_result(0.95), satellite_evidence())

    assert decision.agreement is FireDetectionMLRuleAgreement.ML_STRONGER
    assert decision.final_status is FireDetectionStatus.NO_EVENT  # SHADOW: disagreement does not change outcome


def test_agreement_classification_uses_configured_threshold():
    policy_strict = make_policy(FireDetectionDecisionMode.SHADOW, classification_threshold=0.9)
    rule = rule_decision(FireDetectionStatus.NO_EVENT, 0.0)

    decision = policy_strict.decide(rule, ml_result(0.6), satellite_evidence())

    # 0.6 < 0.9 threshold -> ML classified as "no fire" -> agreement, not ML_STRONGER
    assert decision.agreement is FireDetectionMLRuleAgreement.AGREE_NO_FIRE
