"""Task 9B: the AI audit fields round-trip through the FireEvent ML-assessment row; legacy rows are unaffected."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.models import SatelliteHotspot
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_event_status import FireEventStatus
from src.repositories.fire_event_repository import FireEventRepository

NOW = datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc)


def make_event(repository):
    hotspots = SatelliteHotspotRepository(session_factory=repository._session_factory)
    hotspots.save_hotspot(SatelliteHotspot(latitude=32.7, longitude=35.0, detected_at=NOW, confidence="n", satellite="NOAA-20", instrument="VIIRS"))
    hotspot_id = max(h.id for h in hotspots.get_recent_hotspots(NOW, 60))
    return repository.create_event(
        FireEvent(latitude=32.7, longitude=35.0, detected_at=NOW, updated_at=NOW, status=FireEventStatus.SUSPECTED,
                  detection_confidence=0.6, methodology="t", methodology_version="1"),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),))


def ai_assessment(fire_event_id, probability=0.61, status=FireDetectionStatus.SUSPECTED, passes=1):
    return FireEventMLAssessment(
        fire_event_id=fire_event_id, decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5, rule_status=FireDetectionStatus.SUSPECTED,
        rule_confidence=0.6, ml_available=True, ml_probability=probability, ml_model_name="fire_detection_hgb_v5",
        ml_model_version="5.0", ml_feature_schema_version="v5", ml_failure_reason=None,
        agreement=FireDetectionMLRuleAgreement.AGREE_FIRE, updated_at=NOW, policy_version="ai_hybrid_policy_v5.0",
        policy_status=status, history_available=passes > 1, satellite_pass_count=passes, current_satellite_pixel_count=2)


def legacy_assessment(fire_event_id):
    return FireEventMLAssessment(
        fire_event_id=fire_event_id, decision_mode=FireDetectionDecisionMode.SHADOW, rule_status=FireDetectionStatus.SUSPECTED,
        rule_confidence=0.6, ml_available=True, ml_probability=0.7, ml_model_name="m", ml_model_version="3",
        ml_feature_schema_version="v3", ml_failure_reason=None, agreement=FireDetectionMLRuleAgreement.AGREE_FIRE, updated_at=NOW)


def test_an_ai_assessment_round_trips_with_every_audit_field(sqlite_session_factory):
    repository = FireEventRepository(session_factory=sqlite_session_factory)
    event = make_event(repository)

    repository.upsert_ml_assessment(event.id, ai_assessment(event.id))

    assert repository.get_ml_assessment(event.id) == ai_assessment(event.id)
    assert repository.get_ml_assessments_for_events([event.id])[event.id] == ai_assessment(event.id)


def test_the_latest_ai_assessment_replaces_the_previous_one_in_place(sqlite_session_factory):
    repository = FireEventRepository(session_factory=sqlite_session_factory)
    event = make_event(repository)
    repository.upsert_ml_assessment(event.id, ai_assessment(event.id, 0.61, passes=1))

    repository.upsert_ml_assessment(event.id, ai_assessment(event.id, 0.55, FireDetectionStatus.SUSPECTED, passes=3))

    latest = repository.get_ml_assessment(event.id)
    assert latest.ml_probability == pytest.approx(0.55) and latest.satellite_pass_count == 3 and latest.history_available is True


def test_a_legacy_assessment_carries_no_ai_fields(sqlite_session_factory):
    repository = FireEventRepository(session_factory=sqlite_session_factory)
    event = make_event(repository)

    repository.upsert_ml_assessment(event.id, legacy_assessment(event.id))

    row = repository.get_ml_assessment(event.id)
    assert row == legacy_assessment(event.id)
    assert (row.policy_version, row.policy_status, row.history_available, row.satellite_pass_count) == (None, None, None, None)


def test_a_mixed_batch_of_legacy_and_ai_rows_reads_correctly(sqlite_session_factory):
    repository = FireEventRepository(session_factory=sqlite_session_factory)
    first, second = make_event(repository), make_event(repository)
    repository.upsert_ml_assessment(first.id, legacy_assessment(first.id))
    repository.upsert_ml_assessment(second.id, ai_assessment(second.id))

    batch = repository.get_ml_assessments_for_events([first.id, second.id])

    assert batch[first.id].policy_version is None and batch[second.id].policy_version == "ai_hybrid_policy_v5.0"


def test_ai_fields_are_validated():
    with pytest.raises(ValueError):
        replace(ai_assessment(1), satellite_pass_count=-1)
    with pytest.raises(ValueError):
        replace(ai_assessment(1), policy_status="suspected")
