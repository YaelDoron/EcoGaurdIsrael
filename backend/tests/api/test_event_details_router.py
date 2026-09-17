"""API tests for `GET /api/v1/fire-events/{fire_event_id}/details` (Epic 6, US 6.2, Task 3).

Uses FastAPI's dependency_overrides to replace EventDetailsService with a
fake returning controlled `EventDetailsResult` data - no real DB/service
call happens in these tests.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_event_details_service
from src.api.routers.fire_events import fire_events_router
from src.api.schemas.event_details import (
    BaselineComparisonResponse,
    CurrentResponsePlanResponse,
    EventDetailsResult,
    FireEventSummaryResponse,
    FirefightingResourceResponse,
    FireStationResponse,
    ResponseActionResponse,
    ResponseTargetResponse,
    SeverityAssessmentResponse,
    SpreadPredictionCellResponse,
    SpreadPredictionResponse,
)
from src.database.connection import DatabaseConfigurationError
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType

DETECTED_AT = datetime(2026, 9, 17, 13, 20, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=8)
ASSESSED_AT = DETECTED_AT + timedelta(minutes=7)
PREDICTED_AT = DETECTED_AT + timedelta(minutes=9)
GENERATED_AT = DETECTED_AT + timedelta(minutes=10)
AS_OF = DETECTED_AT + timedelta(minutes=40)

ENDPOINT = "/api/v1/fire-events/{fire_event_id}/details"


class FakeEventDetailsService:
    def __init__(self, result: EventDetailsResult | None):
        self._result = result
        self.calls: list[int] = []

    def get_event_details(self, fire_event_id: int, *, as_of=None):
        self.calls.append(fire_event_id)
        return self._result


class RaisingEventDetailsService:
    def __init__(self, exc: Exception):
        self._exc = exc

    def get_event_details(self, fire_event_id: int, *, as_of=None):
        raise self._exc


def make_fire_event(**overrides) -> FireEventSummaryResponse:
    values = dict(
        fire_event_id=12,
        status=FireEventStatus.CONFIRMED,
        latitude=32.731,
        longitude=35.046,
        detection_confidence=0.91,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        methodology="detector",
        methodology_version="1.0",
    )
    values.update(overrides)
    return FireEventSummaryResponse(**values)


def make_severity(**overrides) -> SeverityAssessmentResponse:
    values = dict(
        assessment_id=44,
        status=FireSeverityAssessmentStatus.VALID,
        score=81.4,
        level=FireSeverityLevel.CRITICAL,
        assessed_at=ASSESSED_AT,
    )
    values.update(overrides)
    return SeverityAssessmentResponse(**values)


def make_spread_prediction(**overrides) -> SpreadPredictionResponse:
    values = dict(
        horizon_minutes=30,
        status=FireSpreadPredictionStatus.VALID,
        predicted_at=PREDICTED_AT,
        cells=[
            SpreadPredictionCellResponse(
                latitude=32.74,
                longitude=35.05,
                spread_probability=0.6,
                spread_risk_score=60.0,
                reached_step=1,
                reached_minutes=5,
            )
        ],
    )
    values.update(overrides)
    return SpreadPredictionResponse(**values)


def make_target(**overrides) -> ResponseTargetResponse:
    values = dict(
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=1.0,
        prediction_horizon_minutes=None,
    )
    values.update(overrides)
    return ResponseTargetResponse(**values)


def make_station(**overrides) -> FireStationResponse:
    values = dict(
        station_id="S1",
        name="Central Station",
        latitude=32.0,
        longitude=34.8,
        station_type="urban",
        address=None,
    )
    values.update(overrides)
    return FireStationResponse(**values)


def make_resource(**overrides) -> FirefightingResourceResponse:
    values = dict(resource_id="R1", station_id="S1", status=ResourceStatus.AVAILABLE)
    values.update(overrides)
    return FirefightingResourceResponse(**values)


def make_plan(**overrides) -> CurrentResponsePlanResponse:
    values = dict(
        plan_id=77,
        generated_at=GENERATED_AT,
        methodology="ga",
        methodology_version="1.0",
        plan_score=90.0,
        coverage_score=1.0,
        average_eta_seconds=120.0,
        actions=[
            ResponseActionResponse(
                resource_id="R1",
                station_id="S1",
                response_target_id=1,
                target_type="active_fire",
                target_priority=1.0,
                eta_seconds=120.0,
                route_distance_meters=500.0,
                node_path=[1, 2, 3],
            )
        ],
        uncovered_target_ids=[],
        baseline_comparison=None,
    )
    values.update(overrides)
    return CurrentResponsePlanResponse(**values)


def make_result(**overrides) -> EventDetailsResult:
    values = dict(
        as_of=AS_OF,
        fire_event=make_fire_event(),
        severity=None,
        danger=None,
        spread_predictions=[],
        targets=[],
        stations=[],
        resources=[],
        current_response_plan=None,
    )
    values.update(overrides)
    return EventDetailsResult(**values)


def client_for(service, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_event_details_service] = lambda: service
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def endpoint_for(fire_event_id: int) -> str:
    return ENDPOINT.format(fire_event_id=fire_event_id)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_full_details():
    result = make_result(
        severity=make_severity(),
        spread_predictions=[make_spread_prediction()],
        targets=[make_target()],
        stations=[make_station()],
        resources=[make_resource()],
        current_response_plan=make_plan(),
    )
    client = client_for(FakeEventDetailsService(result))

    response = client.get(endpoint_for(12))

    assert response.status_code == 200
    body = response.json()
    assert body["fire_event"]["fire_event_id"] == 12
    assert body["severity"]["assessment_id"] == 44
    assert body["danger"] is None
    assert len(body["spread_predictions"]) == 1
    assert len(body["targets"]) == 1
    assert len(body["stations"]) == 1
    assert len(body["resources"]) == 1
    assert body["current_response_plan"]["plan_id"] == 77


def test_service_is_called_with_the_path_fire_event_id():
    service = FakeEventDetailsService(make_result())
    client = client_for(service)

    client.get(endpoint_for(42))

    assert service.calls == [42]


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


def test_missing_fire_event_returns_404():
    client = client_for(FakeEventDetailsService(None))

    response = client.get(endpoint_for(999))

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Current-state spread contract, visible at the HTTP boundary too
# ---------------------------------------------------------------------------


def test_non_valid_spread_prediction_serializes_with_empty_cells():
    result = make_result(
        spread_predictions=[
            make_spread_prediction(status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, cells=[])
        ]
    )
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    prediction = body["spread_predictions"][0]
    assert prediction["status"] == "insufficient_data"
    assert prediction["cells"] == []


# ---------------------------------------------------------------------------
# Missing optional sections serialize as null / empty list
# ---------------------------------------------------------------------------


def test_missing_optional_sections_serialize_as_null_or_empty():
    client = client_for(FakeEventDetailsService(make_result()))

    body = client.get(endpoint_for(12)).json()

    assert body["severity"] is None
    assert body["danger"] is None
    assert body["current_response_plan"] is None
    assert body["spread_predictions"] == []
    assert body["targets"] == []
    assert body["stations"] == []
    assert body["resources"] == []


# ---------------------------------------------------------------------------
# Timezone / ISO-8601 serialization
# ---------------------------------------------------------------------------


def test_timestamps_serialize_with_timezone_information():
    result = make_result(severity=make_severity(), spread_predictions=[make_spread_prediction()])
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    for value in (
        body["as_of"],
        body["fire_event"]["detected_at"],
        body["fire_event"]["updated_at"],
        body["severity"]["assessed_at"],
        body["spread_predictions"][0]["predicted_at"],
    ):
        assert value.endswith("Z") or "+" in value[-6:]
        assert parse_dt(value).tzinfo is not None


# ---------------------------------------------------------------------------
# Enum serialization
# ---------------------------------------------------------------------------


def test_status_and_level_serialize_as_stable_lowercase_strings():
    result = make_result(
        fire_event=make_fire_event(status=FireEventStatus.SUSPECTED),
        severity=make_severity(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None),
    )
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    assert body["fire_event"]["status"] == "suspected"
    assert body["severity"]["status"] == "insufficient_data"
    assert "FireEventStatus" not in str(body)
    assert "FireSeverityAssessmentStatus" not in str(body)


# ---------------------------------------------------------------------------
# Safe failure
# ---------------------------------------------------------------------------


def test_service_failure_returns_5xx_without_leaking_internals():
    secret_exc = DatabaseConfigurationError("DATABASE_URL is not configured.")
    client = client_for(RaisingEventDetailsService(secret_exc), raise_server_exceptions=False)

    response = client.get(endpoint_for(12))

    assert response.status_code >= 500
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "DatabaseConfigurationError"):
        assert leaked not in body_text


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_router_source_calls_no_write_operation():
    import inspect

    details_route = next(
        route for route in fire_events_router.routes if route.path.endswith("/{fire_event_id}/details")
    )
    source = inspect.getsource(details_route.endpoint)
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

    assert "/api/v1/fire-events/{fire_event_id}/details" in schema["paths"]
    get_op = schema["paths"]["/api/v1/fire-events/{fire_event_id}/details"]["get"]
    assert "200" in get_op["responses"]
    response_schema_ref = get_op["responses"]["200"]["content"]["application/json"]["schema"]
    assert "EventDetailsResult" in str(response_schema_ref)
