"""Task 9C: an AI detection result (FireEvent + evidence refs + ML assessment) is stored atomically.

Task 9B's known failure: the FireEvent was committed first and the assessment written second, so a failing assessment write left
an event behind while detect() reported failure. The repository now commits both in ONE transaction.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.agents.analysis import FireDetectionAgent
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.database.base import Base
from src.ml.fire_detection.fire_detection_ai_hybrid_runtime_v5 import FireDetectionAIHybridClassifierV5
from src.models import SatelliteHotspot
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.exceptions import FireEventRepositoryError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService

T0 = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)
LAT, LON = 32.731, 35.046
SUSPECTED, CONFIRMED = FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED


class FailingAssessmentRepository(FireEventRepository):
    """A real repository whose assessment write fails on demand (after the event/evidence writes were issued)."""

    fail_assessment = False

    def _upsert_ml_assessment_in_session(self, session, fire_event_id, assessment):
        if self.fail_assessment:
            raise FireEventRepositoryError("injected assessment failure")
        return super()._upsert_ml_assessment_in_session(session, fire_event_id, assessment)


class ScriptedRuntime:
    def __init__(self, *probabilities):
        self.probabilities = list(probabilities)

    def ensure_loaded(self):
        return SimpleNamespace(model_name="fire_detection_hgb_v5", model_version="5.0", feature_schema_version="v5",
                               policy_version="ai_hybrid_policy_v5.0")

    def predict_probability(self, features):
        return self.probabilities.pop(0)


class Stack:
    def __init__(self, session_factory, *probabilities):
        self.satellites = SatelliteHotspotRepository(session_factory=session_factory)
        self.news = NewsRepository(session_factory=session_factory)
        self.events = FailingAssessmentRepository(session_factory=session_factory)
        self.agent = FireDetectionAgent(
            evidence_service=FireDetectionEvidenceService(satellite_repository=self.satellites, news_repository=self.news),
            calculator=FireDetectionCalculator(), fire_event_repository=self.events,
            decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5,
            ai_classifier=FireDetectionAIHybridClassifierV5(ScriptedRuntime(*probabilities)))

    def hotspot(self, moment, dlat=0.0):
        self.satellites.save_hotspot(SatelliteHotspot(latitude=LAT + dlat, longitude=LON, detected_at=moment, confidence="n", frp=8.0,
                                                      brightness=330.0, satellite="NOAA-20", instrument="VIIRS", day_night="D"))

    def detect(self, moment):
        return self.agent.detect(moment + timedelta(minutes=5))


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    try:
        yield sessionmaker(bind=engine, expire_on_commit=False)
    finally:
        engine.dispose()


def snapshot(stack, event_id):
    stored = stack.events.get_by_id(event_id)
    return stored.event, stack.events.get_evidence_refs(event_id), stack.events.get_ml_assessment(event_id)


# --- the previous behaviour, characterised --------------------------------------------------------------------------------------


def test_the_old_two_step_write_left_an_event_behind_when_the_assessment_failed(session_factory):
    """Why the fix was needed: separate create_event + upsert_ml_assessment calls are two commits."""
    stack = Stack(session_factory)
    stack.hotspot(T0)
    hotspot_id = max(h.id for h in stack.satellites.get_recent_hotspots(T0 + timedelta(hours=1), 600))
    event = stack.events.create_event(
        FireEvent(latitude=LAT, longitude=LON, detected_at=T0, updated_at=T0, status=SUSPECTED, detection_confidence=0.6,
                  methodology="t", methodology_version="1"),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),))
    stack.events.fail_assessment = True
    with pytest.raises(FireEventRepositoryError):
        stack.events.upsert_ml_assessment(event.id, FireEventMLAssessment(
            fire_event_id=event.id, decision_mode=FireDetectionDecisionMode.SHADOW, rule_status=FireDetectionStatus.SUSPECTED,
            rule_confidence=0.6, ml_available=False, ml_probability=None, ml_model_name=None, ml_model_version=None,
            ml_feature_schema_version=None, ml_failure_reason=None, agreement=FireDetectionMLRuleAgreement.ML_UNAVAILABLE, updated_at=T0))
    assert stack.events.get_by_id(event.id) is not None  # the orphan the two-step write produced


# --- new event ---------------------------------------------------------------------------------------------------------------


def test_a_failed_assessment_write_leaves_no_new_event_and_no_evidence_association(session_factory):
    stack = Stack(session_factory, 0.61)
    stack.hotspot(T0)
    stack.events.fail_assessment = True

    result = stack.detect(T0)

    assert result.success is False and "rolled back" in result.error_message and "ai_hybrid_v5" in result.error_message
    assert stack.events.get_recent(10) == ()  # the FireEvent was NOT committed
    assert stack.events.get_active_fire_event_ids() == ()
    from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB

    with session_factory() as session:
        assert session.query(FireEventSatelliteEvidenceDB).count() == 0  # nor its evidence association


def test_a_successful_detection_commits_event_evidence_and_assessment_together(session_factory):
    stack = Stack(session_factory, 0.61)
    stack.hotspot(T0)

    result = stack.detect(T0)

    (event_id,) = result.event_ids
    event, refs, assessment = snapshot(stack, event_id)
    assert result.success and event.status is SUSPECTED and len(refs) == 1
    assert assessment.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5 and assessment.ml_probability == pytest.approx(0.61)


def test_a_retry_after_a_rolled_back_failure_succeeds_cleanly(session_factory):
    stack = Stack(session_factory, 0.61, 0.61)
    stack.hotspot(T0)
    stack.events.fail_assessment = True
    assert stack.detect(T0).success is False
    stack.events.fail_assessment = False

    result = stack.detect(T0)

    assert result.success and result.events_created == 1 and len(stack.events.get_recent(10)) == 1


# --- existing event: promotion + evidence + assessment ------------------------------------------------------------------------------


def test_a_failed_assessment_write_does_not_leave_a_partially_promoted_event(session_factory):
    stack = Stack(session_factory, 0.61, 0.90)
    stack.hotspot(T0)
    (event_id,) = stack.detect(T0).event_ids
    before = snapshot(stack, event_id)
    assert before[0].status is SUSPECTED
    stack.hotspot(T0 + timedelta(hours=3))
    stack.hotspot(T0 + timedelta(hours=3), dlat=0.002)  # two current pixels: the AI would confirm
    stack.events.fail_assessment = True

    result = stack.detect(T0 + timedelta(hours=3))

    assert result.success is False and "rolled back" in result.error_message
    assert snapshot(stack, event_id) == before  # SUSPECTED, same confidence, same refs, same assessment row


def test_the_same_promotion_commits_atomically_when_the_assessment_can_be_stored(session_factory):
    stack = Stack(session_factory, 0.61, 0.90)
    stack.hotspot(T0)
    (event_id,) = stack.detect(T0).event_ids
    stack.hotspot(T0 + timedelta(hours=3))
    stack.hotspot(T0 + timedelta(hours=3), dlat=0.002)

    result = stack.detect(T0 + timedelta(hours=3))

    event, refs, assessment = snapshot(stack, event_id)
    assert result.success and event.status is CONFIRMED and event.detection_confidence == pytest.approx(0.90)
    assert len(refs) == 3 and assessment.policy_status is FireDetectionStatus.CONFIRMED and assessment.satellite_pass_count == 2


def test_a_weak_result_on_an_existing_event_is_also_all_or_nothing(session_factory):
    stack = Stack(session_factory, 0.70, 0.20)
    stack.hotspot(T0)
    (event_id,) = stack.detect(T0).event_ids
    before = snapshot(stack, event_id)
    stack.hotspot(T0 + timedelta(hours=3))
    stack.events.fail_assessment = True

    assert stack.detect(T0 + timedelta(hours=3)).success is False

    assert snapshot(stack, event_id) == before  # the new pass was not attached without its assessment


# --- repository-level contract -------------------------------------------------------------------------------------------------------


def test_the_atomic_repository_methods_validate_before_writing(session_factory):
    stack = Stack(session_factory)
    with pytest.raises(FireEventRepositoryError):
        stack.events.create_event_with_ml_assessment(
            FireEvent(latitude=LAT, longitude=LON, detected_at=T0, updated_at=T0, status=SUSPECTED, detection_confidence=0.6,
                      methodology="t", methodology_version="1"), (), lambda fire_event_id: None)
    with pytest.raises(FireEventRepositoryError):
        stack.events.update_event_with_ml_assessment(999, event=None, new_evidence=(), assessment=None)
    assert stack.events.get_recent(10) == ()


# --- negative candidates ---------------------------------------------------------------------------------------------------------------


def test_a_no_event_candidate_writes_nothing_at_all(session_factory):
    """Only candidates tied to a persisted FireEvent get a persisted assessment; a negative one has the in-memory trace only."""
    from src.database.models.fire_event_db import FireEventDB
    from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB

    stack = Stack(session_factory, 0.20)
    stack.hotspot(T0)

    result = stack.detect(T0)

    assert result.success and result.no_event_count == 1 and result.candidate_assessments[0].event_id is None
    with session_factory() as session:
        assert session.query(FireEventDB).count() == 0 and session.query(FireEventMLAssessmentDB).count() == 0
