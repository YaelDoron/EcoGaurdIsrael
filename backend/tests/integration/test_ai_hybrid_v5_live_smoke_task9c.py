"""Task 9C live smoke against the configured (migrated) Neon demo database.

Proves, on real PostgreSQL: (1) rule_only / shadow / hybrid still work against the migrated schema and never write AI fields,
(2) ai_hybrid_v5 with the REAL approved artifact persists FireEvent + evidence + AI assessment atomically with every audit column,
(3) an assessment failure rolls the whole AI result back. Uses unique far-away coordinates and a fixed past timestamp; every row it
creates is deleted afterwards (by id). Requires DATABASE_URL and the AI-columns migration.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.agents.analysis import FireDetectionAgent
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.config.settings import settings
from src.database.connection import get_engine
from src.models import SatelliteHotspot
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_assessment import FireDetectionMLAssessment
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event_status import FireEventStatus
from src.repositories.exceptions import FireEventRepositoryError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection import FireDetectionEvidenceService

pytestmark = pytest.mark.integration

MOMENT = datetime(2026, 3, 1, 3, 0, tzinfo=timezone.utc)
LAT, LON = 30.11111, 34.55555  # far from every demo location


@pytest.fixture(autouse=True)
def _require_database_url():
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon test.")


@pytest.fixture
def cleanup():
    """Delete everything this test created (FireEvents near the test coordinates, their rows, and the test hotspots)."""
    def wipe():
        with get_engine().begin() as connection:
            ids = [row[0] for row in connection.execute(
                text("SELECT id FROM fire_events WHERE ABS(latitude - :lat) < 0.05 AND ABS(longitude - :lon) < 0.05"), {"lat": LAT, "lon": LON})]
            for fire_event_id in ids:
                for table in ("fire_event_ml_assessments", "fire_event_satellite_evidence", "fire_event_news_evidence"):
                    connection.execute(text(f"DELETE FROM {table} WHERE fire_event_id = :id"), {"id": fire_event_id})
                connection.execute(text("DELETE FROM fire_events WHERE id = :id"), {"id": fire_event_id})
            connection.execute(text("DELETE FROM satellite_hotspots WHERE ABS(latitude - :lat) < 0.05 AND ABS(longitude - :lon) < 0.05"),
                               {"lat": LAT, "lon": LON})
    wipe()
    yield
    wipe()


class FakeV3Classifier:
    def assess(self, evidence):
        return FireDetectionMLAssessment(available=True, probability=0.9, model_name="live-smoke", model_version="1",
                                         feature_schema_version="v3", failure_reason=None)


def _stack(**agent_kwargs):
    satellites, news, events = SatelliteHotspotRepository(), NewsRepository(), FireEventRepository()
    agent = FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news),
        calculator=FireDetectionCalculator(), fire_event_repository=events, satellite_repository=satellites, news_repository=news,
        **agent_kwargs)
    return satellites, events, agent


def _hotspot(satellites, moment=MOMENT, dlat=0.0):
    satellites.save_hotspot(SatelliteHotspot(latitude=LAT + dlat, longitude=LON, detected_at=moment, confidence="n", frp=8.0,
                                             brightness=330.0, satellite="NOAA-20", instrument="VIIRS", day_night="N"))


@pytest.mark.parametrize("mode", [FireDetectionDecisionMode.RULE_ONLY, FireDetectionDecisionMode.SHADOW, FireDetectionDecisionMode.HYBRID])
def test_legacy_modes_work_on_the_migrated_schema_and_never_write_ai_fields(cleanup, mode):
    satellites, events, agent = _stack(
        decision_policy=FireDetectionHybridPolicy(mode=mode, ml_suspect_threshold=0.7), ml_classifier=FakeV3Classifier())
    _hotspot(satellites)

    result = agent.detect(MOMENT + timedelta(minutes=5))

    assert result.success, result.error_message
    (event_id,) = result.event_ids
    assert events.get_by_id(event_id).event.status is FireEventStatus.SUSPECTED  # a lone hotspot: rule decision
    row = events.get_ml_assessment(event_id)
    if mode is FireDetectionDecisionMode.RULE_ONLY:
        assert row is None
    else:
        assert row.decision_mode is mode and row.policy_version is None and row.satellite_pass_count is None
        with get_engine().connect() as connection:  # the AI columns really are NULL in PostgreSQL
            values = connection.execute(text(
                "SELECT policy_version, policy_status, history_available, satellite_pass_count, current_satellite_pixel_count "
                "FROM fire_event_ml_assessments WHERE fire_event_id = :id"), {"id": event_id}).one()
        assert tuple(values) == (None, None, None, None, None)


def test_ai_hybrid_v5_with_the_real_artifact_persists_event_and_ai_assessment_on_postgres(cleanup):
    satellites, events, agent = _stack(decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5)
    _hotspot(satellites)

    result = agent.detect(MOMENT + timedelta(minutes=5))

    assert result.success, result.error_message
    (event_id,) = result.event_ids
    row = events.get_ml_assessment(event_id)
    assert row.decision_mode is FireDetectionDecisionMode.AI_HYBRID_V5 and 0.0 <= row.ml_probability <= 1.0
    assert row.policy_version == "ai_hybrid_policy_v5.0" and row.policy_status in (FireDetectionStatus.SUSPECTED, FireDetectionStatus.CONFIRMED)
    assert (row.history_available, row.satellite_pass_count, row.current_satellite_pixel_count) == (False, 1, 1)
    with get_engine().connect() as connection:
        types = dict(connection.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'fire_event_ml_assessments' "
            "AND column_name IN ('policy_version','policy_status','history_available','satellite_pass_count','current_satellite_pixel_count')")).all())
    assert types == {"policy_version": "character varying", "policy_status": "character varying", "history_available": "boolean",
                     "satellite_pass_count": "integer", "current_satellite_pixel_count": "integer"}


def test_a_failed_assessment_write_rolls_the_whole_ai_result_back_on_postgres(cleanup):
    class Failing(FireEventRepository):
        def _upsert_ml_assessment_in_session(self, session, fire_event_id, assessment):
            raise FireEventRepositoryError("injected assessment failure")

    satellites, _, _ = _stack(decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5)
    news = NewsRepository()
    events = Failing()
    agent = FireDetectionAgent(
        evidence_service=FireDetectionEvidenceService(satellite_repository=satellites, news_repository=news),
        calculator=FireDetectionCalculator(), fire_event_repository=events, decision_mode=FireDetectionDecisionMode.AI_HYBRID_V5)
    _hotspot(satellites)

    result = agent.detect(MOMENT + timedelta(minutes=5))

    assert result.success is False and "rolled back" in result.error_message
    with get_engine().connect() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM fire_events WHERE ABS(latitude - :lat) < 0.05 AND ABS(longitude - :lon) < 0.05"),
                                   {"lat": LAT, "lon": LON}).scalar_one()
    assert count == 0  # no orphan FireEvent
