"""Live integration test for ActiveFireEventsService (Epic 6, US 6.1, Task 2) against Neon.

Exercises the real repository/database layer end to end - persists a
SUSPECTED, a CONFIRMED, and a RESOLVED FireEvent (plus a severity assessment
for one of them) and verifies the service returns only the active two, with
severity linked to the correct event. No FIRMS/IMS/Copernicus/agent access
is required - only this project's own Neon/PostgreSQL database.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

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
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "us6.1-task2-integration"
DETECTED_AT = datetime(2026, 3, 3, 3, 3, 3, tzinfo=timezone.utc)
LATITUDE = 30.12345
LONGITUDE = 34.56789


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
    """Persist a FireEvent with the given final status.

    create_event only accepts SUSPECTED/CONFIRMED (matching Fire Detection's
    own creation contract - see FireEventRepository.create_event); a RESOLVED/
    DISMISSED event is created SUSPECTED and then transitioned via
    update_event, mirroring how the real Fire Detection -> lifecycle flow
    would retire an event.
    """
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


def test_active_events_returned_with_correct_severity_linkage_against_neon():
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

    service = ActiveFireEventsService(
        fire_event_repository=fire_event_repository,
        fire_severity_assessment_repository=severity_repository,
    )
    result = service.get_active_events(as_of=DETECTED_AT + timedelta(hours=1))

    # get_active_events() is intentionally unscoped (like
    # FireEventRepository.get_active_fire_event_ids()), so a shared dev
    # database may contain other active events - assert our known ids
    # behave correctly rather than requiring an exact-set match.
    returned_ids = {item.fire_event_id for item in result.items}
    assert {suspected_id, confirmed_id} <= returned_ids
    assert resolved_id not in returned_ids

    by_id = {item.fire_event_id: item for item in result.items}
    assert by_id[suspected_id].severity is not None
    assert by_id[suspected_id].severity.assessment_id == saved_severity.assessment_id
    assert by_id[suspected_id].severity.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert by_id[confirmed_id].severity is None
