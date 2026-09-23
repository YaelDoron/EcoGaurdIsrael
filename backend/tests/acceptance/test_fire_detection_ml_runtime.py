"""Acceptance coverage for Task 5 runtime ML integration (AT-ML-1..6).

Uses real SQLite-backed repositories (like test_fire_detection_user_story_2_2.py)
so the full evidence -> rule -> ML -> hybrid policy -> FireEvent persistence
path is exercised end to end, but injects a deterministic fake ML classifier
so scenarios ("ML predicts high", "ML predicts low", "ML unavailable") are
fully controlled rather than depending on the real model's actual output.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.analysis import FireDetectionAgent
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.models import FireEventStatus, SatelliteHotspot, WildfireReport
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
CARMEL_LATITUDE = 32.731
CARMEL_LONGITUDE = 35.046


class FakeMLClassifier:
    """Deterministic ML stand-in for acceptance scenarios."""

    def __init__(self, assessment: FireDetectionMLAssessment | None = None, sequence=None):
        self._assessment = assessment
        self._sequence = list(sequence) if sequence is not None else None
        self.calls = 0

    def assess(self, evidence):
        self.calls += 1
        if self._sequence is not None:
            return self._sequence.pop(0)
        return self._assessment


def ml_probability(value: float) -> FireDetectionMLAssessment:
    return FireDetectionMLAssessment(
        available=True,
        probability=value,
        model_name="fire_detection_logistic_v3",
        model_version="3.0",
        feature_schema_version="v3",
        failure_reason=None,
    )


def ml_unavailable(reason: str = "ML model unavailable.") -> FireDetectionMLAssessment:
    return FireDetectionMLAssessment(
        available=False,
        probability=None,
        model_name=None,
        model_version=None,
        feature_schema_version=None,
        failure_reason=reason,
    )


def make_agent(sqlite_session_factory, decision_policy, ml_classifier) -> FireDetectionAgent:
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news_repository = NewsRepository(session_factory=sqlite_session_factory)
    evidence_service = FireDetectionEvidenceService(
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    return FireDetectionAgent(
        evidence_service=evidence_service,
        calculator=FireDetectionCalculator(),
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
        satellite_repository=satellite_repository,
        news_repository=news_repository,
        decision_policy=decision_policy,
        ml_classifier=ml_classifier,
    )


def save_satellite(sqlite_session_factory, *, detected_at=AS_OF, confidence="n") -> int:
    repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    repository.save_hotspot(
        SatelliteHotspot(
            latitude=CARMEL_LATITUDE,
            longitude=CARMEL_LONGITUDE,
            detected_at=detected_at,
            confidence=confidence,
            satellite="SIM-NOAA-20",
            instrument="VIIRS",
        )
    )
    return repository.get_recent_hotspots(as_of=detected_at + timedelta(minutes=1), lookback_minutes=10)[0].id


def save_news(sqlite_session_factory, *, source_url, published_at=AS_OF) -> int:
    repository = NewsRepository(session_factory=sqlite_session_factory)
    repository.save_report(
        WildfireReport(
            source_url=source_url,
            source_feed="Acceptance",
            title="Wildfire reported",
            summary="Smoke and flames reported.",
            location_name="Acceptance Location",
            latitude=CARMEL_LATITUDE,
            longitude=CARMEL_LONGITUDE,
            published_at=published_at,
            fetched_at=published_at + timedelta(minutes=1),
        )
    )
    return repository.get_recent_reports(as_of=published_at + timedelta(minutes=2), lookback_minutes=10)[0].id


def get_stored_event(sqlite_session_factory, event_id: int):
    return FireEventRepository(session_factory=sqlite_session_factory).get_by_id(event_id)


def get_ml_assessment(sqlite_session_factory, event_id: int):
    return FireEventRepository(session_factory=sqlite_session_factory).get_ml_assessment(event_id)


SHADOW_POLICY = FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.SHADOW, ml_suspect_threshold=0.70)
HYBRID_POLICY = FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.HYBRID, ml_suspect_threshold=0.70)
RULE_ONLY_POLICY = FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.RULE_ONLY)


# --- AT-ML-1: Shadow agreement ---


def test_at_ml_1_shadow_agreement(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    save_news(sqlite_session_factory, source_url="https://acceptance.example/ml1", published_at=AS_OF + timedelta(minutes=20))
    ml_classifier = FakeMLClassifier(assessment=ml_probability(0.9))

    result = make_agent(sqlite_session_factory, SHADOW_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    event_id = result.event_ids[0]
    stored = get_stored_event(sqlite_session_factory, event_id)
    assert stored.event.status is FireEventStatus.CONFIRMED  # rule confirms fire

    assessment = get_ml_assessment(sqlite_session_factory, event_id)
    assert assessment is not None
    assert assessment.ml_available is True
    assert assessment.ml_probability == pytest.approx(0.9)
    assert assessment.agreement is FireDetectionMLRuleAgreement.AGREE_FIRE


# --- AT-ML-2: Shadow disagreement ---


def test_at_ml_2_shadow_disagreement(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    save_news(sqlite_session_factory, source_url="https://acceptance.example/ml2", published_at=AS_OF + timedelta(minutes=20))
    ml_classifier = FakeMLClassifier(assessment=ml_probability(0.1))  # low ML, disagrees

    result = make_agent(sqlite_session_factory, SHADOW_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    event_id = result.event_ids[0]
    stored = get_stored_event(sqlite_session_factory, event_id)
    assert stored.event.status is FireEventStatus.CONFIRMED  # fire remains confirmed despite ML disagreement
    assert stored.event.detection_confidence == pytest.approx(0.80)

    assessment = get_ml_assessment(sqlite_session_factory, event_id)
    assert assessment.agreement is FireDetectionMLRuleAgreement.RULE_STRONGER  # disagreement visible


# --- AT-ML-3: ML failure ---


def test_at_ml_3_ml_failure_still_detects_via_rule(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    save_news(sqlite_session_factory, source_url="https://acceptance.example/ml3", published_at=AS_OF + timedelta(minutes=20))
    ml_classifier = FakeMLClassifier(assessment=ml_unavailable())

    result = make_agent(sqlite_session_factory, SHADOW_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    event_id = result.event_ids[0]
    stored = get_stored_event(sqlite_session_factory, event_id)
    assert stored.event.status is FireEventStatus.CONFIRMED

    assessment = get_ml_assessment(sqlite_session_factory, event_id)
    assert assessment.ml_available is False
    assert assessment.agreement is FireDetectionMLRuleAgreement.ML_UNAVAILABLE


# --- AT-ML-4: New evidence recalculates ML ---


def test_at_ml_4_new_evidence_recalculates_ml_probability(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    ml_classifier = FakeMLClassifier(sequence=[ml_probability(0.3)])
    first = make_agent(sqlite_session_factory, SHADOW_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=5))
    event_id = first.event_ids[0]
    first_assessment = get_ml_assessment(sqlite_session_factory, event_id)
    assert first_assessment.ml_probability == pytest.approx(0.3)

    save_news(sqlite_session_factory, source_url="https://acceptance.example/ml4", published_at=AS_OF + timedelta(minutes=20))
    # candidate-level check + combined-evidence reevaluation both return the new probability
    ml_classifier_2 = FakeMLClassifier(sequence=[ml_probability(0.95), ml_probability(0.95)])
    second = make_agent(sqlite_session_factory, SHADOW_POLICY, ml_classifier_2).detect(as_of=AS_OF + timedelta(minutes=25))

    assert second.events_updated == 1
    second_assessment = get_ml_assessment(sqlite_session_factory, event_id)
    assert second_assessment.ml_probability == pytest.approx(0.95)  # recalculated, not stale


# --- AT-ML-5: RULE_ONLY reproduces historical behavior ---


def test_at_ml_5_rule_only_reproduces_historical_behavior(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="n", detected_at=AS_OF)
    save_news(sqlite_session_factory, source_url="https://acceptance.example/ml5", published_at=AS_OF + timedelta(minutes=20))
    ml_classifier = FakeMLClassifier(assessment=ml_probability(0.99))  # must never be consulted

    result = make_agent(sqlite_session_factory, RULE_ONLY_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=25))

    assert result.success is True
    event_id = result.event_ids[0]
    stored = get_stored_event(sqlite_session_factory, event_id)
    assert stored.event.status is FireEventStatus.CONFIRMED
    assert stored.event.detection_confidence == pytest.approx(0.80)
    assert ml_classifier.calls == 0
    assert get_ml_assessment(sqlite_session_factory, event_id) is None  # no ML trace row at all


# --- AT-ML-6: Hybrid conservative escalation ---


def test_at_ml_6_hybrid_escalation_reaches_at_most_suspected(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="low", detected_at=AS_OF)  # rule alone: NO_EVENT
    ml_classifier = FakeMLClassifier(assessment=ml_probability(0.95))  # far above the 0.70 suspect threshold

    result = make_agent(sqlite_session_factory, HYBRID_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=5))

    assert result.success is True
    assert result.no_event_count == 0
    assert result.events_created == 1
    event_id = result.event_ids[0]
    stored = get_stored_event(sqlite_session_factory, event_id)
    assert stored.event.status is FireEventStatus.SUSPECTED  # escalated, but never CONFIRMED by ML alone

    assessment = get_ml_assessment(sqlite_session_factory, event_id)
    assert assessment.agreement is FireDetectionMLRuleAgreement.ML_STRONGER


def test_at_ml_6_hybrid_no_escalation_below_threshold(sqlite_session_factory):
    save_satellite(sqlite_session_factory, confidence="low", detected_at=AS_OF)
    ml_classifier = FakeMLClassifier(assessment=ml_probability(0.5))  # below the 0.70 suspect threshold

    result = make_agent(sqlite_session_factory, HYBRID_POLICY, ml_classifier).detect(as_of=AS_OF + timedelta(minutes=5))

    assert result.success is True
    assert result.no_event_count == 1
    assert result.events_created == 0
