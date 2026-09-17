"""Live integration test for `GET /api/v1/fire-events/active` (Epic 6, US 6.1, Task 3).

Exercises the real stack end to end - HTTP (FastAPI TestClient, no
dependency overrides) -> router -> ActiveFireEventsService -> repositories
-> Neon/PostgreSQL. Persists a SUSPECTED, a CONFIRMED, and a RESOLVED
FireEvent (plus a severity assessment for one of them) and verifies the
endpoint returns only the active two, with severity linked to the correct
event. No FIRMS/IMS/Copernicus/agent access is required - only this
project's own Neon/PostgreSQL database.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.api.app import create_app
from src.calculators.fire_detection.fire_detection_config import FIRE_DETECTION_METHODOLOGY_NAME
from src.config.settings import settings
from src.database.connection import get_engine, init_db
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    SatelliteHotspot,
)
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "us6.1-task3-api-integration"
DETECTED_AT = datetime(2026, 4, 4, 4, 4, 4, tzinfo=timezone.utc)
LATITUDE = 29.98765
LONGITUDE = 34.98765

ENDPOINT = "/api/v1/fire-events/active"


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
                "DELETE FROM fire_severity_assessments WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                ")"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM fire_event_satellite_evidence WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                ")"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM fire_events WHERE methodology_version = :methodology_version"),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM satellite_hotspots WHERE latitude = :latitude AND longitude = :longitude"),
            {"latitude": LATITUDE, "longitude": LONGITUDE},
        )


def _create_event(
    fire_event_repository: FireEventRepository,
    satellite_repository: SatelliteHotspotRepository,
    *,
    status: FireEventStatus,
    offset_minutes: int,
) -> int:
    """Persist a FireEvent with the given final status (see the Task 2
    integration test for why RESOLVED/DISMISSED events go through
    create_event(SUSPECTED) + update_event rather than create_event directly)."""
    detected_at = DETECTED_AT + timedelta(minutes=offset_minutes)
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=detected_at,
            confidence="h",
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(
        as_of=detected_at + timedelta(minutes=1),
        lookback_minutes=2,
    )[0].id
    creation_status = status if status in (FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED) else FireEventStatus.SUSPECTED
    saved = fire_event_repository.create_event(
        FireEvent(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=detected_at,
            updated_at=detected_at,
            status=creation_status,
            detection_confidence=0.6,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
        ),
        (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    if status is not creation_status:
        fire_event_repository.update_event(
            saved.id,
            FireEvent(
                latitude=LATITUDE,
                longitude=LONGITUDE,
                detected_at=detected_at,
                updated_at=detected_at + timedelta(minutes=1),
                status=status,
                detection_confidence=0.6,
                methodology=FIRE_DETECTION_METHODOLOGY_NAME,
                methodology_version=METHODOLOGY_VERSION,
            ),
        )
    return saved.id


def test_active_fire_events_endpoint_against_neon():
    fire_event_repository = FireEventRepository()
    satellite_repository = SatelliteHotspotRepository()
    severity_repository = FireSeverityAssessmentRepository()

    suspected_id = _create_event(
        fire_event_repository, satellite_repository, status=FireEventStatus.SUSPECTED, offset_minutes=0
    )
    confirmed_id = _create_event(
        fire_event_repository, satellite_repository, status=FireEventStatus.CONFIRMED, offset_minutes=5
    )
    resolved_id = _create_event(
        fire_event_repository, satellite_repository, status=FireEventStatus.RESOLVED, offset_minutes=10
    )

    saved_severity = severity_repository.save_assessment(
        FireSeverityAssessment(
            fire_event_id=suspected_id,
            assessed_at=DETECTED_AT + timedelta(minutes=1),
            status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
            score=None,
            level=None,
            methodology="severity-model",
            methodology_version="1.0",
        ),
        weather_observation_ids=(),
        satellite_hotspot_ids=(),
        selected_frp_hotspot_id=None,
    )

    client = TestClient(create_app())
    response = client.get(ENDPOINT)

    assert response.status_code == 200
    body = response.json()

    items_by_id = {item["fire_event_id"]: item for item in body["items"]}
    # Unscoped endpoint (matches ActiveFireEventsService/get_active_fire_event_ids
    # convention) - a shared dev database may contain other active events, so
    # assert our known ids behave correctly rather than requiring an exact-set match.
    assert suspected_id in items_by_id
    assert confirmed_id in items_by_id
    assert resolved_id not in items_by_id

    assert items_by_id[suspected_id]["status"] == "suspected"
    assert items_by_id[suspected_id]["severity"] is not None
    assert items_by_id[suspected_id]["severity"]["assessment_id"] == saved_severity.assessment_id
    assert items_by_id[suspected_id]["severity"]["status"] == "insufficient_data"
    assert items_by_id[suspected_id]["severity"]["score"] is None
    assert items_by_id[suspected_id]["severity"]["level"] is None

    assert items_by_id[confirmed_id]["status"] == "confirmed"
    assert items_by_id[confirmed_id]["severity"] is None
