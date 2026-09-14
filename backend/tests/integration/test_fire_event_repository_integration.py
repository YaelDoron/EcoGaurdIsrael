"""Live integration test for FireEvent persistence against Neon."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.config.settings import settings
from src.database.connection import get_engine, get_session, init_db
from src.models import FireEvent, FireEventStatus, FireEvidenceRef, FireEvidenceType, SatelliteHotspot, WildfireReport
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

pytestmark = pytest.mark.integration

TEST_SOURCE_URL = "https://example.com/ecoguard-integration/fire-event-task-3"
DETECTED_AT = datetime(2026, 2, 22, 2, 22, 22, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=15)
LATITUDE = 31.23456
LONGITUDE = 35.98765


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
                "SELECT id FROM fire_events "
                "WHERE methodology = :methodology AND detected_at = :detected_at"
                ")"
            ),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": DETECTED_AT},
        )
        connection.execute(
            text(
                "DELETE FROM fire_event_news_evidence "
                "WHERE fire_event_id IN ("
                "SELECT id FROM fire_events "
                "WHERE methodology = :methodology AND detected_at = :detected_at"
                ")"
            ),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": DETECTED_AT},
        )
        connection.execute(
            text(
                "DELETE FROM fire_events "
                "WHERE methodology = :methodology AND detected_at = :detected_at"
            ),
            {"methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": DETECTED_AT},
        )
        connection.execute(
            text("DELETE FROM wildfire_reports WHERE source_url = :source_url"),
            {"source_url": TEST_SOURCE_URL},
        )
        connection.execute(
            text(
                "DELETE FROM satellite_hotspots "
                "WHERE detected_at = :detected_at AND latitude = :latitude AND longitude = :longitude"
            ),
            {"detected_at": DETECTED_AT, "latitude": LATITUDE, "longitude": LONGITUDE},
        )


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        status=FireEventStatus.SUSPECTED,
        detection_confidence=0.6,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def _create_source_evidence() -> tuple[int, int]:
    satellite_repository = SatelliteHotspotRepository()
    news_repository = NewsRepository()
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=DETECTED_AT,
            confidence="h",
            frp=22.5,
            satellite="N20",
        )
    )
    news_repository.save_report(
        WildfireReport(
            source_url=TEST_SOURCE_URL,
            source_feed="Integration",
            title="Integration wildfire report",
            summary="Integration test fire report.",
            location_name="Integration Ridge",
            latitude=LATITUDE,
            longitude=LONGITUDE,
            published_at=DETECTED_AT,
            fetched_at=DETECTED_AT + timedelta(minutes=2),
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(
        as_of=DETECTED_AT + timedelta(minutes=1),
        lookback_minutes=10,
    )[0].id
    report_id = news_repository.get_recent_reports(
        as_of=DETECTED_AT + timedelta(minutes=3),
        lookback_minutes=10,
    )[0].id
    return hotspot_id, report_id


def test_fire_event_repository_round_trip_against_neon():
    hotspot_id, report_id = _create_source_evidence()
    repository = FireEventRepository()

    saved = repository.create_event(
        make_event(),
        (
            FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),
            FireEvidenceRef(FireEvidenceType.NEWS, report_id),
        ),
    )
    found = repository.get_by_id(saved.id)

    assert found.id == saved.id
    assert found.event == saved.event
    assert found.supporting_evidence == (
        FireEvidenceRef(FireEvidenceType.NEWS, report_id),
        FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),
    )

    with get_session() as session:
        relation_values = session.execute(
            text(
                "SELECT "
                "(SELECT satellite_hotspot_id FROM fire_event_satellite_evidence WHERE fire_event_id = :event_id) "
                "AS satellite_hotspot_id, "
                "(SELECT wildfire_report_id FROM fire_event_news_evidence WHERE fire_event_id = :event_id) "
                "AS wildfire_report_id"
            ),
            {"event_id": saved.id},
        ).one()

    assert relation_values.satellite_hotspot_id == hotspot_id
    assert relation_values.wildfire_report_id == report_id


def test_fire_event_active_lookup_and_update_against_neon():
    hotspot_id, _ = _create_source_evidence()
    repository = FireEventRepository()
    saved = repository.create_event(make_event(), (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),))

    matched = repository.find_matching_active_event(LATITUDE, LONGITUDE, UPDATED_AT + timedelta(hours=1))
    assert matched.id == saved.id

    updated = repository.update_event(
        saved.id,
        make_event(status=FireEventStatus.CONFIRMED, detection_confidence=0.85),
    )
    assert updated.event.status is FireEventStatus.CONFIRMED
    assert updated.event.detection_confidence == pytest.approx(0.85)
