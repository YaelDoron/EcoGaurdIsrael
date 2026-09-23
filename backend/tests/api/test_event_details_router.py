"""API tests for `GET /api/v1/fire-events/{fire_event_id}/details` (Epic 6, US 6.2, Task 3).

Uses FastAPI's dependency_overrides to replace EventDetailsService with a
fake returning controlled `EventDetailsResult` data - no real DB/service
call happens in these tests.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_event_details_service
from src.api.routers.fire_events import fire_events_router
from src.api.schemas.event_details import (
    BaselineComparisonResponse,
    CurrentResponsePlanResponse,
    DetectionEvidenceResponse,
    EventDetailsResult,
    FireEventMLAssessmentResponse,
    FireEventSummaryResponse,
    FirefightingResourceResponse,
    FireStationResponse,
    NewsEvidenceResponse,
    ResponseActionResponse,
    ResponseTargetResponse,
    SatelliteEvidenceResponse,
    SeverityAssessmentResponse,
    SpreadPredictionCellResponse,
    SpreadPredictionResponse,
    StationAllocationResponse,
    StationSummaryResponse,
)
from src.database.connection import DatabaseConfigurationError
from src.models.fire_detection_decision_mode import FireDetectionDecisionMode
from src.models.fire_detection_ml_rule_agreement import FireDetectionMLRuleAgreement
from src.models.fire_detection_status import FireDetectionStatus
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


def make_ml_assessment(**overrides) -> FireEventMLAssessmentResponse:
    values = dict(
        available=True,
        mode=FireDetectionDecisionMode.SHADOW,
        rule_status=FireDetectionStatus.CONFIRMED,
        rule_confidence=0.80,
        model_score=0.75,
        agreement=FireDetectionMLRuleAgreement.AGREE_FIRE,
        model_name="fire_detection_logistic_v3",
        model_version="3.0",
        feature_schema_version="v3",
        failure_reason=None,
        updated_at=UPDATED_AT,
    )
    values.update(overrides)
    return FireEventMLAssessmentResponse(**values)


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


def make_satellite_evidence(**overrides) -> SatelliteEvidenceResponse:
    values = dict(
        id=501,
        detected_at=DETECTED_AT,
        latitude=32.7,
        longitude=35.0,
        confidence="high",
        frp=15.0,
        brightness=310.0,
        satellite="Terra",
        instrument="MODIS",
        day_night="D",
    )
    values.update(overrides)
    return SatelliteEvidenceResponse(**values)


def make_news_evidence(**overrides) -> NewsEvidenceResponse:
    values = dict(
        id=701,
        title="Blaze reported near reserve",
        summary="A wildfire was reported near the nature reserve.",
        source="haaretz",
        observed_at=DETECTED_AT,
        location_name="Modiin",
        latitude=31.9,
        longitude=35.0,
    )
    values.update(overrides)
    return NewsEvidenceResponse(**values)


def make_detection_evidence(**overrides) -> DetectionEvidenceResponse:
    values = dict(satellite=[], news=[])
    values.update(overrides)
    return DetectionEvidenceResponse(**values)


def make_station_allocation(**overrides) -> StationAllocationResponse:
    values = dict(resource_id="R1", fire_event_id=12, response_plan_id=77)
    values.update(overrides)
    return StationAllocationResponse(**values)


def make_station_summary(**overrides) -> StationSummaryResponse:
    values = dict(
        station_id="S1",
        total_resources=4,
        available=2,
        assigned_status=1,
        unavailable=1,
        current_global_plan_allocations=[],
    )
    values.update(overrides)
    return StationSummaryResponse(**values)


def make_result(**overrides) -> EventDetailsResult:
    values = dict(
        as_of=AS_OF,
        fire_event=make_fire_event(),
        severity=None,
        danger=None,
        detection_evidence=make_detection_evidence(),
        spread_predictions=[],
        targets=[],
        stations=[],
        resources=[],
        station_summaries=[],
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
        detection_evidence=make_detection_evidence(
            satellite=[make_satellite_evidence()], news=[make_news_evidence()]
        ),
        spread_predictions=[make_spread_prediction()],
        targets=[make_target()],
        stations=[make_station()],
        resources=[make_resource()],
        station_summaries=[make_station_summary(current_global_plan_allocations=[make_station_allocation()])],
        current_response_plan=make_plan(),
    )
    client = client_for(FakeEventDetailsService(result))

    response = client.get(endpoint_for(12))

    assert response.status_code == 200
    body = response.json()
    assert body["fire_event"]["fire_event_id"] == 12
    assert body["severity"]["assessment_id"] == 44
    assert body["danger"] is None
    assert len(body["detection_evidence"]["satellite"]) == 1
    assert body["detection_evidence"]["satellite"][0]["id"] == 501
    assert body["detection_evidence"]["satellite"][0]["satellite"] == "Terra"
    assert len(body["detection_evidence"]["news"]) == 1
    assert body["detection_evidence"]["news"][0]["id"] == 701
    assert body["detection_evidence"]["news"][0]["source"] == "haaretz"
    assert len(body["spread_predictions"]) == 1
    assert len(body["targets"]) == 1
    assert len(body["stations"]) == 1
    assert len(body["resources"]) == 1
    assert body["current_response_plan"]["plan_id"] == 77
    assert len(body["station_summaries"]) == 1
    summary = body["station_summaries"][0]
    assert summary["station_id"] == "S1"
    assert summary["total_resources"] == 4
    assert summary["available"] == 2
    assert summary["assigned_status"] == 1
    assert summary["unavailable"] == 1
    assert len(summary["current_global_plan_allocations"]) == 1
    allocation = summary["current_global_plan_allocations"][0]
    assert allocation["resource_id"] == "R1"
    assert allocation["fire_event_id"] == 12
    assert allocation["response_plan_id"] == 77


def test_service_is_called_with_the_path_fire_event_id():
    service = FakeEventDetailsService(make_result())
    client = client_for(service)

    client.get(endpoint_for(42))

    assert service.calls == [42]


# ---------------------------------------------------------------------------
# Detection evidence
# ---------------------------------------------------------------------------


def test_detection_evidence_is_serialized_with_timezone_aware_timestamps():
    result = make_result(
        detection_evidence=make_detection_evidence(
            satellite=[make_satellite_evidence()], news=[make_news_evidence()]
        )
    )
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    satellite = body["detection_evidence"]["satellite"][0]
    news = body["detection_evidence"]["news"][0]
    for value in (satellite["detected_at"], news["observed_at"]):
        assert value.endswith("Z") or "+" in value[-6:]
        assert parse_dt(value).tzinfo is not None


def test_no_detection_evidence_serializes_as_empty_lists():
    client = client_for(FakeEventDetailsService(make_result()))

    body = client.get(endpoint_for(12)).json()

    assert body["detection_evidence"]["satellite"] == []
    assert body["detection_evidence"]["news"] == []


# ---------------------------------------------------------------------------
# ML assessment (ML Task 6: API exposure only)
# ---------------------------------------------------------------------------


def test_shadow_ml_assessment_is_exposed_with_all_fields():
    result = make_result(ml_assessment=make_ml_assessment())
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

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
    # ML did not control the FireEvent's own final status.
    assert body["fire_event"]["status"] == "confirmed"


def test_rule_stronger_ml_assessment_is_exposed():
    result = make_result(
        fire_event=make_fire_event(status=FireEventStatus.SUSPECTED),
        ml_assessment=make_ml_assessment(
            rule_status=FireDetectionStatus.SUSPECTED,
            rule_confidence=0.60,
            model_score=0.18,
            agreement=FireDetectionMLRuleAgreement.RULE_STRONGER,
        ),
    )
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    ml = body["ml_assessment"]
    assert ml["rule_confidence"] == pytest.approx(0.60)
    assert ml["model_score"] == pytest.approx(0.18)
    assert ml["agreement"] == "rule_stronger"
    assert body["fire_event"]["status"] == "suspected"


def test_no_ml_assessment_row_serializes_as_null():
    """Legacy FireEvents (created before ML integration) must still return
    a valid 200 response, with ml_assessment as null - not an error, and
    not a fabricated zero-valued object."""
    result = make_result(ml_assessment=None)
    client = client_for(FakeEventDetailsService(result))

    response = client.get(endpoint_for(12))

    assert response.status_code == 200
    assert response.json()["ml_assessment"] is None


def test_ml_unavailable_does_not_substitute_zero():
    result = make_result(
        ml_assessment=make_ml_assessment(
            available=False,
            model_score=None,
            model_name=None,
            model_version=None,
            feature_schema_version=None,
            failure_reason="ML model artifact could not be loaded.",
            agreement=FireDetectionMLRuleAgreement.ML_UNAVAILABLE,
        )
    )
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    ml = body["ml_assessment"]
    assert ml["available"] is False
    assert ml["model_score"] is None
    assert ml["model_score"] != 0
    assert ml["agreement"] == "ml_unavailable"
    assert ml["failure_reason"] == "ML model artifact could not be loaded."


def test_rule_only_event_with_no_ml_row_remains_valid():
    """RULE_ONLY mode never writes a FireEventMLAssessment row."""
    result = make_result(ml_assessment=None)
    client = client_for(FakeEventDetailsService(result))

    response = client.get(endpoint_for(12))

    assert response.status_code == 200
    body = response.json()
    assert body["ml_assessment"] is None
    assert body["fire_event"]["fire_event_id"] == 12


def test_hybrid_mode_serializes_without_special_case_failure():
    """Schema supports HYBRID even though it is not the current default."""
    result = make_result(ml_assessment=make_ml_assessment(mode=FireDetectionDecisionMode.HYBRID))
    client = client_for(FakeEventDetailsService(result))

    response = client.get(endpoint_for(12))

    assert response.status_code == 200
    assert response.json()["ml_assessment"]["mode"] == "hybrid"


def test_model_version_and_feature_schema_version_are_exposed_unchanged():
    result = make_result(
        ml_assessment=make_ml_assessment(model_version="3.0", feature_schema_version="v3")
    )
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    assert body["ml_assessment"]["model_version"] == "3.0"
    assert body["ml_assessment"]["feature_schema_version"] == "v3"


def test_ml_assessment_updated_at_serializes_with_timezone_information():
    result = make_result(ml_assessment=make_ml_assessment())
    client = client_for(FakeEventDetailsService(result))

    body = client.get(endpoint_for(12)).json()

    updated_at = body["ml_assessment"]["updated_at"]
    assert updated_at.endswith("Z") or "+" in updated_at[-6:]
    assert parse_dt(updated_at).tzinfo is not None


def test_ml_assessment_does_not_leak_internal_diagnostic_text():
    """Defense in depth at the API test layer: even if a caller passed a
    raw exception-shaped string through, this test documents the contract
    that failure_reason must read as a short, safe, generic category -
    real sanitization happens in EventDetailsService (see its own tests)."""
    result = make_result(
        ml_assessment=make_ml_assessment(
            available=False, model_score=None, failure_reason="ML model artifact could not be loaded."
        )
    )
    client = client_for(FakeEventDetailsService(result))

    body_text = client.get(endpoint_for(12)).text

    for leaked in ("Traceback", "joblib", "FileNotFoundError", ".pkl", ":\\", "/etc/"):
        assert leaked not in body_text


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
    assert body["station_summaries"] == []
    assert body["detection_evidence"] == {"satellite": [], "news": []}


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


def test_ml_assessment_schema_is_registered_as_an_optional_nested_object():
    app = create_app()

    schema = app.openapi()

    components = schema["components"]["schemas"]
    assert "FireEventMLAssessmentResponse" in components
    ml_schema = components["FireEventMLAssessmentResponse"]
    for field in (
        "available",
        "mode",
        "rule_status",
        "rule_confidence",
        "model_score",
        "agreement",
        "model_version",
        "feature_schema_version",
        "failure_reason",
        "updated_at",
    ):
        assert field in ml_schema["properties"]

    event_details_schema = components["EventDetailsResult"]
    ml_field = event_details_schema["properties"]["ml_assessment"]
    # Optional: anyOf [ref, null] (Pydantic's Optional[...] shape), not a bare required ref.
    assert any(option.get("type") == "null" for option in ml_field.get("anyOf", []))
    assert "ml_assessment" not in event_details_schema.get("required", [])
