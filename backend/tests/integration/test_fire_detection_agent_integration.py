"""Live integration test for FireDetectionAgent against Neon persistence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.agents.analysis.fire_detection_agent import FireDetectionAgent
from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_decision_policy import FireDetectionHybridPolicy
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.calculators.fire_detection.fire_detection_config import FIRE_DETECTION_METHODOLOGY_NAME
from src.config.settings import settings
from src.database.connection import get_engine, init_db
from src.models import FireEventStatus, SatelliteHotspot, WildfireReport
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.fire_detection.fire_detection_evidence_service import FireDetectionEvidenceService

pytestmark = pytest.mark.integration

SOURCE_URL = "https://example.com/ecoguard-integration/fire-detection-agent"
SATELLITE_TIME = datetime(2026, 3, 3, 3, 0, tzinfo=timezone.utc)
NEWS_TIME = SATELLITE_TIME + timedelta(minutes=20)
AS_OF = NEWS_TIME + timedelta(minutes=5)
LATITUDE = 32.731
LONGITUDE = 35.046


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url) -> None:
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM fire_event_satellite_evidence "
                "WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology = :methodology AND detected_at = :detected_at"
                ")"
            ),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": SATELLITE_TIME},
        )
        connection.execute(
            text(
                "DELETE FROM fire_event_news_evidence "
                "WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology = :methodology AND detected_at = :detected_at"
                ")"
            ),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": SATELLITE_TIME},
        )
        connection.execute(
            text("DELETE FROM fire_events WHERE methodology = :methodology AND detected_at = :detected_at"),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": SATELLITE_TIME},
        )
        connection.execute(text("DELETE FROM wildfire_reports WHERE source_url = :source_url"), {"source_url": SOURCE_URL})
        connection.execute(
            text(
                "DELETE FROM satellite_hotspots "
                "WHERE detected_at = :detected_at AND latitude = :latitude AND longitude = :longitude"
            ),
            {"detected_at": SATELLITE_TIME, "latitude": LATITUDE, "longitude": LONGITUDE},
        )


def make_agent() -> FireDetectionAgent:
    satellite_repository = SatelliteHotspotRepository()
    news_repository = NewsRepository()
    evidence_service = FireDetectionEvidenceService(
        satellite_repository=satellite_repository,
        news_repository=news_repository,
    )
    return FireDetectionAgent(
        evidence_service=evidence_service,
        calculator=FireDetectionCalculator(),
        fire_event_repository=FireEventRepository(),
        satellite_repository=satellite_repository,
        news_repository=news_repository,
        # Pinned (Task 9C): these rule-behaviour tests must not depend on FIRE_DETECTION_DECISION_MODE in the environment.
        decision_policy=FireDetectionHybridPolicy(mode=FireDetectionDecisionMode.SHADOW),
    )


def test_fire_detection_agent_creates_upgrades_and_reprocesses_without_duplicate_event():
    satellite_repository = SatelliteHotspotRepository()
    news_repository = NewsRepository()
    event_repository = FireEventRepository()

    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=SATELLITE_TIME,
            confidence="n",
            satellite="N20",
        )
    )
    first = make_agent().detect(as_of=SATELLITE_TIME + timedelta(minutes=5))

    assert first.success is True
    assert first.events_created == 1
    assert first.events_updated == 0
    event_id = first.event_ids[0]
    stored = event_repository.get_by_id(event_id)
    assert stored.event.status is FireEventStatus.SUSPECTED
    assert stored.event.detection_confidence == pytest.approx(0.6)

    news_repository.save_report(
        WildfireReport(
            source_url=SOURCE_URL,
            source_feed="Integration",
            title="Wildfire reported near Carmel",
            summary="Smoke observed near forest.",
            location_name="Carmel",
            latitude=LATITUDE,
            longitude=LONGITUDE,
            published_at=NEWS_TIME,
            fetched_at=NEWS_TIME + timedelta(minutes=1),
        )
    )
    second = make_agent().detect(as_of=AS_OF)

    assert second.success is True
    assert second.events_created == 0
    assert second.events_updated == 1
    assert second.event_ids == (event_id,)
    upgraded = event_repository.get_by_id(event_id)
    assert upgraded.event.status is FireEventStatus.CONFIRMED
    assert upgraded.event.detection_confidence == pytest.approx(0.8)
    assert len(upgraded.supporting_evidence) == 2
    assert {ref.evidence_type.value for ref in upgraded.supporting_evidence} == {"news", "satellite"}

    third = make_agent().detect(as_of=AS_OF)

    assert third.success is True
    assert third.events_created == 0
    assert third.events_updated == 0
    assert third.event_ids == (event_id,)
