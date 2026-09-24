"""Task 5B acceptance: multi-overpass event evidence history (full stack on SQLite).

Real SQLite repositories, the real evidence service, calculator and FireDetectionAgent; only the ML
classifier is a deterministic stand-in. No wall-clock waiting: every time is a simulated timestamp.

These tests were first written against the UNCHANGED code to reproduce two failures:

  1. multi-overpass: a hotspot 3 hours after an event's first evidence matches the same FireEvent, but
     the agent re-evaluated ALL attached evidence as one FireDetectionCandidate, which refuses evidence
     more than 60 minutes apart -> ValueError -> detect() failed and attached nothing;
  2. hybrid re-evaluation: an event created by ML escalation of a rule-NO_EVENT candidate was re-scored by
     the rule alone on the next cycle -> "unexpectedly evaluated as NO_EVENT" -> detect() failed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

import pytest

from src.agents.analysis import FireDetectionAgent
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.models import FireEventStatus, SatelliteHotspot
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
LATITUDE = 32.731
LONGITUDE = 35.046


class FakeMLClassifier:
    def __init__(self, probability: float):
        self.probability = probability

    def assess(self, evidence):
        return FireDetectionMLAssessment(
            available=True,
            probability=self.probability,
            model_name="fire_detection_logistic_v3",
            model_version="3.0",
            feature_schema_version="v3",
            failure_reason=None,
        )


def build_agent(session_factory, mode=FireDetectionDecisionMode.RULE_ONLY, ml_probability=0.5):
    satellite_repository = SatelliteHotspotRepository(session_factory=session_factory)
    news_repository = NewsRepository(session_factory=session_factory)
    events = FireEventRepository(session_factory=session_factory)
    policy = FireDetectionHybridPolicy(mode=mode, ml_suspect_threshold=0.70)
    agent = FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(
            satellite_repository=satellite_repository, news_repository=news_repository
        ),
        calculator=FireDetectionCalculator(),
        fire_event_repository=events,
        satellite_repository=satellite_repository,
        news_repository=news_repository,
        decision_policy=policy,
        ml_classifier=None if mode is FireDetectionDecisionMode.RULE_ONLY else FakeMLClassifier(ml_probability),
    )
    agent.classifier = agent._ml_classifier
    return agent, satellite_repository, events


def save_hotspot(repository, detected_at, *, confidence="n", latitude=LATITUDE, longitude=LONGITUDE, frp=None):
    repository.save_hotspot(
        SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            confidence=confidence,
            frp=frp,
            satellite="NOAA-20",
            instrument="VIIRS",
        )
    )


def event_ids(events: FireEventRepository):
    return [stored.id for stored in events.get_recent(50)]


def test_a_hotspot_three_hours_later_updates_the_same_fire_event_and_attaches_its_evidence(sqlite_session_factory, caplog):
    agent, satellites, events = build_agent(sqlite_session_factory)
    save_hotspot(satellites, T0)
    first = agent.detect(T0 + timedelta(minutes=5))
    assert first.success and first.events_created == 1
    (event_id,) = first.event_ids
    assert len(events.get_evidence_refs(event_id)) == 1

    save_hotspot(satellites, T0 + timedelta(hours=3), confidence="h", latitude=LATITUDE + 0.004)
    with caplog.at_level(logging.ERROR):
        second = agent.detect(T0 + timedelta(hours=3, minutes=5))

    assert second.success is True, second.error_message
    assert second.events_created == 0 and second.events_updated == 1
    assert second.event_ids == (event_id,)
    assert event_ids(events) == [event_id]  # still exactly one FireEvent
    assert len(events.get_evidence_refs(event_id)) == 2  # both observation waves are attached


def test_a_rule_no_event_event_created_by_ml_escalation_survives_re_evaluation(sqlite_session_factory):
    agent, satellites, events = build_agent(sqlite_session_factory, FireDetectionDecisionMode.HYBRID, ml_probability=0.9)
    save_hotspot(satellites, T0, confidence="l")  # a lone LOW hotspot: the rule alone says NO_EVENT
    first = agent.detect(T0 + timedelta(minutes=5))
    assert first.success and first.events_created == 1
    (event_id,) = first.event_ids
    assert events.get_by_id(event_id).event.status is FireEventStatus.SUSPECTED

    second = agent.detect(T0 + timedelta(minutes=15))  # same evidence, next cycle

    assert second.success is True, second.error_message
    assert event_ids(events) == [event_id]


# ---------------------------------------------------------------------------
# Wider multi-overpass scenarios (simulated timestamps only)
# ---------------------------------------------------------------------------
from src.services.fire_detection import FireDetectionHistoryService  # noqa: E402


def history_of(session_factory, events, event_id, as_of):
    service = FireDetectionHistoryService(
        FireDetectionEvidenceService(
            satellite_repository=SatelliteHotspotRepository(session_factory=session_factory),
            news_repository=NewsRepository(session_factory=session_factory),
        ),
        events,
    )
    return service.build_history(events.get_by_id(event_id), as_of)


def test_three_overpass_waves_keep_one_event_and_build_a_three_pass_history(sqlite_session_factory):
    agent, satellites, events = build_agent(sqlite_session_factory)
    save_hotspot(satellites, T0, frp=5.0)
    (event_id,) = agent.detect(T0 + timedelta(minutes=5)).event_ids
    for hours, frp in ((3, 10.0), (6, 15.0)):
        save_hotspot(satellites, T0 + timedelta(hours=hours), confidence="h", frp=frp)
        result = agent.detect(T0 + timedelta(hours=hours, minutes=5))
        assert result.success, result.error_message
        assert result.event_ids == (event_id,)

    assert event_ids(events) == [event_id]
    stored = events.get_by_id(event_id)
    assert stored.event.detected_at == T0  # identity/first-seen preserved across waves
    assert stored.event.updated_at == T0 + timedelta(hours=6)

    history = history_of(sqlite_session_factory, events, event_id, T0 + timedelta(hours=6, minutes=5))
    assert history.distinct_satellite_pass_count == 3
    assert history.satellite_observation_span_minutes == pytest.approx(360.0)
    assert history.frp_trend().slope_per_hour == pytest.approx(5.0 / 3.0)  # 5 -> 10 -> 15 FRP over 0 / 3 / 6 h


def test_a_separate_fire_far_away_gets_its_own_event(sqlite_session_factory):
    agent, satellites, events = build_agent(sqlite_session_factory)
    save_hotspot(satellites, T0)
    (first_id,) = agent.detect(T0 + timedelta(minutes=5)).event_ids

    save_hotspot(satellites, T0 + timedelta(hours=3), latitude=LATITUDE + 0.2)  # ~22 km away
    second = agent.detect(T0 + timedelta(hours=3, minutes=5))

    assert second.success and second.events_created == 1 and second.events_updated == 0
    assert len(events.get_evidence_refs(first_id)) == 1  # the first fire's evidence is untouched
    assert len(event_ids(events)) == 2


def test_a_weaker_later_wave_does_not_lower_the_events_confidence_or_status(sqlite_session_factory):
    agent, satellites, events = build_agent(sqlite_session_factory)
    for offset in (0, 2):
        save_hotspot(satellites, T0 + timedelta(minutes=offset), confidence="h")
    (event_id,) = agent.detect(T0 + timedelta(minutes=10)).event_ids
    before = events.get_by_id(event_id).event

    save_hotspot(satellites, T0 + timedelta(hours=3), confidence="n")  # one nominal pixel
    result = agent.detect(T0 + timedelta(hours=3, minutes=5))

    assert result.success, result.error_message
    after = events.get_by_id(event_id).event
    assert after.detection_confidence >= before.detection_confidence
    assert after.status is before.status or before.status is FireEventStatus.SUSPECTED
    assert after.detected_at == before.detected_at


def test_re_running_the_same_cycle_changes_nothing(sqlite_session_factory):
    agent, satellites, events = build_agent(sqlite_session_factory)
    save_hotspot(satellites, T0)
    (event_id,) = agent.detect(T0 + timedelta(minutes=5)).event_ids
    save_hotspot(satellites, T0 + timedelta(hours=3), confidence="h")
    assert agent.detect(T0 + timedelta(hours=3, minutes=5)).events_updated == 1

    again = agent.detect(T0 + timedelta(hours=3, minutes=6))

    assert again.success and again.events_created == 0 and again.events_updated == 0
    assert len(events.get_evidence_refs(event_id)) == 2


def test_hybrid_ml_drop_on_re_evaluation_keeps_the_event_without_crashing(sqlite_session_factory):
    agent, satellites, events = build_agent(sqlite_session_factory, FireDetectionDecisionMode.HYBRID, ml_probability=0.9)
    save_hotspot(satellites, T0, confidence="l")
    (event_id,) = agent.detect(T0 + timedelta(minutes=5)).event_ids
    before = events.get_by_id(event_id).event

    agent.classifier.probability = 0.1  # the ML no longer supports the escalation
    result = agent.detect(T0 + timedelta(minutes=15))

    assert result.success is True, result.error_message
    assert event_ids(events) == [event_id]  # not deleted
    after = events.get_by_id(event_id).event
    assert (after.status, after.detection_confidence, after.latitude, after.longitude) == (
        before.status,
        before.detection_confidence,
        before.latitude,
        before.longitude,
    )


def _fresh_session_factory():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.database.base import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, expire_on_commit=False), engine


def _run_multi_overpass(mode):
    session_factory, engine = _fresh_session_factory()
    try:
        agent, satellites, events = build_agent(session_factory, mode, ml_probability=0.2)
        save_hotspot(satellites, T0)
        assert agent.detect(T0 + timedelta(minutes=5)).success
        save_hotspot(satellites, T0 + timedelta(hours=3), confidence="h")
        assert agent.detect(T0 + timedelta(hours=3, minutes=5)).success
        (stored,) = events.get_recent(10)
        return stored.event.status, round(stored.event.detection_confidence, 9), len(events.get_evidence_refs(stored.id))
    finally:
        engine.dispose()


@pytest.mark.parametrize("mode", [FireDetectionDecisionMode.SHADOW, FireDetectionDecisionMode.HYBRID])
def test_shadow_and_hybrid_match_rule_only_on_a_multi_overpass_fire(mode):
    """ML never changes the outcome of a fire the rule already detects; the modes keep their meaning."""
    assert _run_multi_overpass(mode) == _run_multi_overpass(FireDetectionDecisionMode.RULE_ONLY)
