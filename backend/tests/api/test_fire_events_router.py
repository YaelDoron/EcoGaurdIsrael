"""API tests for `GET /api/v1/fire-events/active` (Epic 6, US 6.1, Task 3).

Uses FastAPI's dependency_overrides to replace ActiveFireEventsService with
a fake returning controlled `ActiveFireEventsResult` data - no real DB/
service call happens in these tests. Real end-to-end coverage (HTTP ->
router -> service -> repositories -> Neon) is in
tests/integration/test_active_fire_events_api_integration.py.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_active_fire_events_service
from src.api.routers.fire_events import fire_events_router
from src.database.connection import DatabaseConfigurationError
from src.models.active_fire_events import (
    ActiveFireEventMLSummary,
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
)
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel

DETECTED_AT = datetime(2026, 9, 17, 13, 20, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=8)
ASSESSED_AT = DETECTED_AT + timedelta(minutes=7)
AS_OF = DETECTED_AT + timedelta(minutes=40)

ENDPOINT = "/api/v1/fire-events/active"


class FakeActiveFireEventsService:
    def __init__(self, result: ActiveFireEventsResult):
        self._result = result
        self.calls = 0

    def get_active_events(self, *, as_of=None):
        self.calls += 1
        return self._result


class RaisingActiveFireEventsService:
    def __init__(self, exc: Exception):
        self._exc = exc

    def get_active_events(self, *, as_of=None):
        raise self._exc


def make_severity(**overrides) -> ActiveFireEventSeveritySummary:
    values = dict(
        assessment_id=44,
        status=FireSeverityAssessmentStatus.VALID,
        score=81.4,
        level=FireSeverityLevel.CRITICAL,
        assessed_at=ASSESSED_AT,
    )
    values.update(overrides)
    return ActiveFireEventSeveritySummary(**values)


def make_event(**overrides) -> ActiveFireEventSummary:
    values = dict(
        fire_event_id=12,
        status=FireEventStatus.CONFIRMED,
        latitude=32.731,
        longitude=35.046,
        detection_confidence=0.91,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        created_at=UPDATED_AT,
        severity=None,
    )
    values.update(overrides)
    return ActiveFireEventSummary(**values)


def client_for(service, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_active_fire_events_service] = lambda: service
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_suspected_and_confirmed_events():
    suspected = make_event(fire_event_id=1, status=FireEventStatus.SUSPECTED)
    confirmed = make_event(fire_event_id=2, status=FireEventStatus.CONFIRMED, severity=make_severity())
    result = ActiveFireEventsResult(as_of=AS_OF, items=(suspected, confirmed))
    client = client_for(FakeActiveFireEventsService(result))

    response = client.get(ENDPOINT)

    assert response.status_code == 200
    body = response.json()
    assert parse_dt(body["as_of"]) == AS_OF
    assert len(body["items"]) == 2
    assert body["items"][0] == {
        "fire_event_id": 1,
        "status": "suspected",
        "latitude": 32.731,
        "longitude": 35.046,
        "detection_confidence": 0.91,
        "detected_at": "2026-09-17T13:20:00Z",
        "updated_at": "2026-09-17T13:28:00Z",
        "created_at": "2026-09-17T13:28:00Z",
        "severity": None,
        "location_name": None,
        "ml_summary": None,
    }
    assert body["items"][1]["fire_event_id"] == 2
    assert body["items"][1]["status"] == "confirmed"
    assert body["items"][1]["severity"]["assessment_id"] == 44


# ---------------------------------------------------------------------------
# Severity serialization
# ---------------------------------------------------------------------------


def test_severity_fields_serialize_correctly():
    event = make_event(severity=make_severity(assessment_id=44, score=81.4, level=FireSeverityLevel.CRITICAL))
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    response = client.get(ENDPOINT)

    severity = response.json()["items"][0]["severity"]
    assert severity["assessment_id"] == 44
    assert severity["status"] == "valid"
    assert severity["score"] == 81.4
    assert severity["level"] == "critical"
    assert parse_dt(severity["assessed_at"]) == ASSESSED_AT


# ---------------------------------------------------------------------------
# Missing severity
# ---------------------------------------------------------------------------


def test_missing_severity_serializes_as_null():
    event = make_event(severity=None)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    response = client.get(ENDPOINT)

    assert response.json()["items"][0]["severity"] is None


# ---------------------------------------------------------------------------
# ML summary serialization (dashboard Active Fire cards, ML Task 7)
# ---------------------------------------------------------------------------


def test_ml_summary_available_serializes_correctly():
    event = make_event(ml_summary=ActiveFireEventMLSummary(available=True, model_score=0.9933510680894274))
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    body = client.get(ENDPOINT).json()

    ml_summary = body["items"][0]["ml_summary"]
    assert ml_summary == {"available": True, "model_score": 0.9933510680894274}


def test_ml_summary_missing_serializes_as_null():
    event = make_event(ml_summary=None)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    body = client.get(ENDPOINT).json()

    assert body["items"][0]["ml_summary"] is None


def test_ml_summary_unavailable_never_substitutes_zero():
    event = make_event(ml_summary=ActiveFireEventMLSummary(available=False, model_score=None))
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    body = client.get(ENDPOINT).json()

    ml_summary = body["items"][0]["ml_summary"]
    assert ml_summary["available"] is False
    assert ml_summary["model_score"] is None


def test_several_active_events_each_keep_their_own_ml_summary():
    first = make_event(fire_event_id=1, ml_summary=ActiveFireEventMLSummary(available=True, model_score=0.2))
    second = make_event(fire_event_id=2, ml_summary=ActiveFireEventMLSummary(available=True, model_score=0.8))
    third = make_event(fire_event_id=3, ml_summary=None)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(first, second, third))
    client = client_for(FakeActiveFireEventsService(result))

    body = client.get(ENDPOINT).json()

    by_id = {item["fire_event_id"]: item for item in body["items"]}
    assert by_id[1]["ml_summary"]["model_score"] == pytest.approx(0.2)
    assert by_id[2]["ml_summary"]["model_score"] == pytest.approx(0.8)
    assert by_id[3]["ml_summary"] is None


# ---------------------------------------------------------------------------
# Empty collection
# ---------------------------------------------------------------------------


def test_empty_collection_returns_200_with_empty_items():
    result = ActiveFireEventsResult(as_of=AS_OF, items=())
    client = client_for(FakeActiveFireEventsService(result))

    response = client.get(ENDPOINT)

    assert response.status_code == 200
    assert response.json() == {"as_of": "2026-09-17T14:00:00Z", "items": []}


# ---------------------------------------------------------------------------
# Ordering preservation
# ---------------------------------------------------------------------------


def test_response_preserves_service_ordering_without_resorting():
    """The service already owns ordering (Task 2); give it a non-id-sorted
    order and confirm the router/mapper does not apply its own sort."""
    first = make_event(fire_event_id=9)
    second = make_event(fire_event_id=1)
    third = make_event(fire_event_id=5)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(first, second, third))
    client = client_for(FakeActiveFireEventsService(result))

    response = client.get(ENDPOINT)

    assert [item["fire_event_id"] for item in response.json()["items"]] == [9, 1, 5]


# ---------------------------------------------------------------------------
# Timezone serialization
# ---------------------------------------------------------------------------


def test_timestamps_serialize_with_timezone_information():
    event = make_event(severity=make_severity())
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    body = client.get(ENDPOINT).json()

    for value in (
        body["as_of"],
        body["items"][0]["detected_at"],
        body["items"][0]["updated_at"],
        body["items"][0]["created_at"],
        body["items"][0]["severity"]["assessed_at"],
    ):
        assert value.endswith("Z") or "+" in value[-6:]
        assert parse_dt(value).tzinfo is not None


# ---------------------------------------------------------------------------
# Enum serialization
# ---------------------------------------------------------------------------


def test_status_and_level_serialize_as_stable_lowercase_strings():
    event = make_event(
        status=FireEventStatus.SUSPECTED,
        severity=make_severity(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None),
    )
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    item = client.get(ENDPOINT).json()["items"][0]

    assert item["status"] == "suspected"
    assert item["severity"]["status"] == "insufficient_data"
    assert "FireEventStatus" not in str(item)
    assert "FireSeverityAssessmentStatus" not in str(item)


# ---------------------------------------------------------------------------
# No ORM/internal leakage
# ---------------------------------------------------------------------------


def test_response_contains_only_defined_dto_fields():
    event = make_event(severity=make_severity())
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))
    client = client_for(FakeActiveFireEventsService(result))

    body = client.get(ENDPOINT).json()

    assert set(body.keys()) == {"as_of", "items"}
    assert set(body["items"][0].keys()) == {
        "fire_event_id",
        "status",
        "latitude",
        "longitude",
        "detection_confidence",
        "detected_at",
        "updated_at",
        "created_at",
        "severity",
        "location_name",
        "ml_summary",
    }
    assert set(body["items"][0]["severity"].keys()) == {
        "assessment_id",
        "status",
        "score",
        "level",
        "assessed_at",
    }
    forbidden = ("_sa_instance_state", "supporting_evidence", "weather_input_ids", "satellite_input_ids")
    body_text = str(body)
    for field in forbidden:
        assert field not in body_text


# ---------------------------------------------------------------------------
# Safe failure
# ---------------------------------------------------------------------------


def test_service_failure_returns_5xx_without_leaking_internals():
    secret_exc = DatabaseConfigurationError("DATABASE_URL is not configured.")
    client = client_for(RaisingActiveFireEventsService(secret_exc), raise_server_exceptions=False)

    response = client.get(ENDPOINT)

    assert response.status_code >= 500
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "DatabaseConfigurationError"):
        assert leaked not in body_text


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_router_source_calls_no_write_operation():
    import inspect

    source = inspect.getsource(fire_events_router.routes[0].endpoint)
    for forbidden in (".save(", ".update(", ".delete(", ".create_event(", "update_event", "Agent"):
        assert forbidden not in source


def test_router_module_does_not_import_agents_or_external_providers():
    forbidden_fragments = (
        "src.agents",
        "src.external",
        "src.simulation",
        "src.calculators",
        "sqlalchemy",
    )
    path = Path("backend/src/api/routers/fire_events.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


def test_endpoint_is_registered_in_openapi_schema():
    app = create_app()

    schema = app.openapi()

    assert "/api/v1/fire-events/active" in schema["paths"]
    get_op = schema["paths"]["/api/v1/fire-events/active"]["get"]
    assert "200" in get_op["responses"]
    response_schema_ref = get_op["responses"]["200"]["content"]["application/json"]["schema"]
    assert "ActiveFireEventsResponse" in str(response_schema_ref)
