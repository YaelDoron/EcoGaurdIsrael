"""Live integration test for `GET /api/v1/fire-events/{fire_event_id}/details` ML exposure (ML Task 6).

Exercises the real stack end to end - HTTP (FastAPI TestClient, no
dependency overrides) -> router -> EventDetailsService -> FireEventRepository
-> Neon/PostgreSQL. Persists a real FireEvent and a real FireEventMLAssessment
row (the exact shape a SHADOW-mode Fire Detection refresh would write) and
verifies the event-details endpoint serializes it correctly. No
FIRMS/IMS/Copernicus/ML-model-artifact access is required - only this
project's own Neon/PostgreSQL database, matching this subsystem's existing
integration-test convention (see test_active_fire_events_api_integration.py).
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
from src.models import FireEvent, FireEventStatus, FireEvidenceRef, FireEvidenceType, SatelliteHotspot
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "ml-task6-event-details-api-integration"
DETECTED_AT = datetime(2026, 4, 5, 5, 5, 5, tzinfo=timezone.utc)
LATITUDE = 30.12345
LONGITUDE = 35.12345

ENDPOINT = "/api/v1/fire-events/{fire_event_id}/details"


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
                "DELETE FROM fire_event_ml_assessments WHERE fire_event_id IN ("
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


def _create_event(fire_event_repository: FireEventRepository, satellite_repository: SatelliteHotspotRepository) -> int:
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=DETECTED_AT,
            confidence="h",
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(
        as_of=DETECTED_AT + timedelta(minutes=1),
        lookback_minutes=2,
    )[0].id
    saved = fire_event_repository.create_event(
        FireEvent(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=DETECTED_AT,
            updated_at=DETECTED_AT,
            status=FireEventStatus.CONFIRMED,
            detection_confidence=0.80,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
        ),
        (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return saved.id


def test_event_details_endpoint_exposes_shadow_ml_assessment_against_neon():
    fire_event_repository = FireEventRepository()
    satellite_repository = SatelliteHotspotRepository()

    fire_event_id = _create_event(fire_event_repository, satellite_repository)
    fire_event_repository.upsert_ml_assessment(
        fire_event_id,
        FireEventMLAssessment(
            fire_event_id=fire_event_id,
            decision_mode=FireDetectionDecisionMode.SHADOW,
            rule_status=FireDetectionStatus.CONFIRMED,
            rule_confidence=0.80,
            ml_available=True,
            ml_probability=0.75,
            ml_model_name="fire_detection_logistic_v3",
            ml_model_version="3.0",
            ml_feature_schema_version="v3",
            ml_failure_reason=None,
            agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
            updated_at=DETECTED_AT,
        ),
    )

    client = TestClient(create_app())
    response = client.get(ENDPOINT.format(fire_event_id=fire_event_id))

    assert response.status_code == 200
    body = response.json()

    assert body["fire_event"]["fire_event_id"] == fire_event_id
    assert body["fire_event"]["status"] == "confirmed"

    ml = body["ml_assessment"]
    assert ml is not None
    assert ml["available"] is True
    assert ml["mode"] == "shadow"
    assert ml["rule_status"] == "confirmed"
    assert ml["rule_confidence"] == pytest.approx(0.80)
    assert ml["model_score"] == pytest.approx(0.75)
    assert ml["agreement"] == "agree_fire"
    assert ml["model_name"] == "fire_detection_logistic_v3"
    assert ml["model_version"] == "3.0"
    assert ml["feature_schema_version"] == "v3"
    assert ml["failure_reason"] is None


def test_event_details_endpoint_returns_null_ml_assessment_when_no_row_exists():
    fire_event_repository = FireEventRepository()
    satellite_repository = SatelliteHotspotRepository()

    fire_event_id = _create_event(fire_event_repository, satellite_repository)

    client = TestClient(create_app())
    response = client.get(ENDPOINT.format(fire_event_id=fire_event_id))

    assert response.status_code == 200
    assert response.json()["ml_assessment"] is None
