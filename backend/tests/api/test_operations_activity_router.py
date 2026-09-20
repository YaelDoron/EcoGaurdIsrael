"""API tests for the Operations Activity detail endpoint (Task A5, Part 22).

Uses FastAPI's dependency_overrides to replace OperationsActivityQueryService
with a fake returning controlled domain read-model data - no real
repository/DB/agent/calculator call happens in these tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_operations_activity_query_service
from src.models.fire_danger_assessment_detail import FireDangerAssessmentDetail
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_report import WildfireReport
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.operations_activity import (
    FireDangerActivityDetail,
    FireEventActivityDetail,
    FireEventActivityDetails,
    FireEventEvidenceRefs,
    FireSeverityActivityDetail,
    FireSeverityActivityDetails,
    GlobalPlanningRunActivityDetail,
    GlobalPlanningRunActivityDetails,
    GlobalPlanningRunMemberSummary,
    NewsReportActivityDetail,
    OperationsActivityLocation,
    OperationsActivityType,
    SatelliteHotspotActivityDetail,
    WeatherConditionsActivityDetail,
    WeatherConditionsActivityDetails,
    WeatherConditionsStationReading,
)
from src.models.satellite_hotspot import SatelliteHotspot

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)


def endpoint(activity_type: str, entity_id) -> str:
    return f"/api/v1/operations/activity/{activity_type}/{entity_id}"


class FakeOperationsActivityQueryService:
    def __init__(self, *, by_key=None):
        self._by_key = by_key or {}
        self.calls: list[tuple] = []

    def get_activity_detail(self, activity_type, entity_id):
        self.calls.append((activity_type, entity_id))
        return self._by_key.get((activity_type, entity_id))


def client_for(service, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_operations_activity_query_service] = lambda: service
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Sample domain object builders
# ---------------------------------------------------------------------------


def make_fire_danger() -> FireDangerActivityDetail:
    fd_detail = FireDangerAssessmentDetail(
        assessment_id=1,
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        assessed_at=ASSESSED_AT,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
        weather_inputs=(),
    )
    return FireDangerActivityDetail(
        entity_id=1,
        occurred_at=ASSESSED_AT,
        title="Fire Danger Assessment — Carmel",
        location=OperationsActivityLocation(latitude=32.731, longitude=35.046),
        details=fd_detail,
    )


def make_fire_danger_insufficient() -> FireDangerActivityDetail:
    fd_detail = FireDangerAssessmentDetail(
        assessment_id=2,
        area_id="area-golan",
        area_name="Golan",
        area_latitude=33.0,
        area_longitude=35.7,
        area_radius_km=5.0,
        status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
        assessed_at=ASSESSED_AT,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
        weather_inputs=(),
    )
    return FireDangerActivityDetail(
        entity_id=2,
        occurred_at=ASSESSED_AT,
        title="Fire Danger Assessment — Golan",
        location=OperationsActivityLocation(latitude=33.0, longitude=35.7),
        details=fd_detail,
    )


def make_satellite() -> SatelliteHotspotActivityDetail:
    hotspot = SatelliteHotspot(
        latitude=32.7,
        longitude=35.0,
        detected_at=datetime(2026, 9, 19, 11, 0),
        confidence="high",
        frp=45.6,
        satellite="N",
    )
    return SatelliteHotspotActivityDetail(
        entity_id=7,
        occurred_at=hotspot.detected_at,
        title="Satellite Hotspot",
        location=OperationsActivityLocation(latitude=32.7, longitude=35.0),
        details=hotspot,
    )


def make_news() -> NewsReportActivityDetail:
    report = WildfireReport(
        source_url="https://example.com/report",
        source_feed="example-feed",
        title="Wildfire spreads near Carmel",
        summary="A wildfire is spreading near the Carmel region.",
        location_name="Carmel",
        latitude=32.7,
        longitude=35.0,
        published_at=ASSESSED_AT,
        fetched_at=ASSESSED_AT,
    )
    return NewsReportActivityDetail(
        entity_id=3,
        occurred_at=ASSESSED_AT,
        title=report.title,
        location=OperationsActivityLocation(latitude=32.7, longitude=35.0),
        details=report,
    )


def make_fire_event() -> FireEventActivityDetail:
    event = FireEvent(
        latitude=32.7,
        longitude=35.0,
        detected_at=ASSESSED_AT,
        updated_at=ASSESSED_AT,
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.87,
        methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version="1.0",
    )
    return FireEventActivityDetail(
        entity_id=42,
        occurred_at=ASSESSED_AT,
        title="Fire Event #42",
        location=OperationsActivityLocation(latitude=32.7, longitude=35.0),
        details=FireEventActivityDetails(
            fire_event=event,
            evidence=FireEventEvidenceRefs(satellite_hotspot_ids=(7,), news_report_ids=(3,)),
            latest_severity=None,
            created_at=ASSESSED_AT,
        ),
    )


def make_fire_severity() -> FireSeverityActivityDetail:
    assessment = FireSeverityAssessment(
        fire_event_id=42,
        assessed_at=ASSESSED_AT,
        status=FireSeverityAssessmentStatus.VALID,
        score=61.25,
        level=FireSeverityLevel.CRITICAL,
        methodology="ECOGUARD_SEVERITY",
        methodology_version="1.0",
    )
    return FireSeverityActivityDetail(
        entity_id=9,
        occurred_at=ASSESSED_AT,
        title="Fire Severity Assessment",
        location=None,
        details=FireSeverityActivityDetails(
            assessment=assessment,
            weather_observation_ids=(101, 102),
            satellite_hotspot_ids=(7,),
            selected_frp_hotspot_id=7,
        ),
    )


def make_global_planning_run() -> GlobalPlanningRunActivityDetail:
    run = GlobalPlanningRun(
        started_at=ASSESSED_AT,
        completed_at=ASSESSED_AT,
        status=GlobalPlanningRunStatus.COMPLETED,
        trigger="weather_event",
        methodology="legacy_per_event_orchestration",
        methodology_version="1.0",
        input_fingerprint="fp-1",
    )
    members = (
        GlobalPlanningRunMemberSummary(
            fire_event_id=101,
            event_order=0,
            result_status=GlobalPlanningRunEventStatus.PLANNED,
            response_plan_id=201,
            severity_level=FireSeverityLevel.HIGH,
            assigned_resources=3,
            coverage_score=80.0,
            average_eta_seconds=320.5,
        ),
        GlobalPlanningRunMemberSummary(
            fire_event_id=102,
            event_order=1,
            result_status=GlobalPlanningRunEventStatus.PLANNED,
            response_plan_id=202,
            severity_level=FireSeverityLevel.MODERATE,
            assigned_resources=2,
            coverage_score=70.0,
            average_eta_seconds=280.0,
        ),
    )
    return GlobalPlanningRunActivityDetail(
        entity_id=5,
        occurred_at=ASSESSED_AT,
        title="Global Response Plan",
        location=None,
        details=GlobalPlanningRunActivityDetails(run=run, members=members),
    )


def make_weather_conditions() -> WeatherConditionsActivityDetail:
    return WeatherConditionsActivityDetail(
        entity_id=1,
        occurred_at=ASSESSED_AT,
        title="Weather Conditions — Carmel",
        location=OperationsActivityLocation(latitude=32.731, longitude=35.046),
        details=WeatherConditionsActivityDetails(
            fire_danger_assessment_id=1,
            area_name="Carmel",
            fire_danger_level=FireDangerLevel.HIGH,
            assessed_at=ASSESSED_AT,
            readings=(
                WeatherConditionsStationReading(
                    station_id=100,
                    station_name="Carmel Station",
                    observation_id=10,
                    observed_at=ASSESSED_AT,
                    temperature=34.0,
                    relative_humidity=19.0,
                    wind_speed=28.0,
                    wind_gust=None,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1 & 2. Each supported activity_type -> 200
# ---------------------------------------------------------------------------


def test_fire_danger_valid_entity_returns_200():
    detail = make_fire_danger()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_DANGER, 1): detail})
    client = client_for(service)

    response = client.get(endpoint("fire_danger", 1))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "fire_danger"
    assert body["entity_id"] == 1
    assert body["details"]["score"] == 42.5


def test_satellite_hotspot_valid_entity_returns_200():
    detail = make_satellite()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.SATELLITE_HOTSPOT, 7): detail})
    client = client_for(service)

    response = client.get(endpoint("satellite_hotspot", 7))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "satellite_hotspot"
    assert body["details"]["confidence"] == "high"


def test_news_report_valid_entity_returns_200():
    detail = make_news()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.NEWS_REPORT, 3): detail})
    client = client_for(service)

    response = client.get(endpoint("news_report", 3))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "news_report"
    assert body["title"] == "Wildfire spreads near Carmel"


def test_fire_event_valid_entity_returns_200():
    detail = make_fire_event()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_EVENT, 42): detail})
    client = client_for(service)

    response = client.get(endpoint("fire_event", 42))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "fire_event"
    assert body["details"]["fire_event_id"] == 42


def test_fire_severity_valid_entity_returns_200():
    detail = make_fire_severity()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_SEVERITY, 9): detail})
    client = client_for(service)

    response = client.get(endpoint("fire_severity", 9))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "fire_severity"
    assert body["details"]["level"] == "critical"


def test_global_planning_run_valid_entity_returns_200():
    detail = make_global_planning_run()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.GLOBAL_PLANNING_RUN, 5): detail})
    client = client_for(service)

    response = client.get(endpoint("global_planning_run", 5))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "global_planning_run"
    assert len(body["details"]["members"]) == 2


def test_weather_conditions_valid_entity_returns_200():
    detail = make_weather_conditions()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.WEATHER_CONDITIONS, 1): detail})
    client = client_for(service)

    response = client.get(endpoint("weather_conditions", 1))

    assert response.status_code == 200
    body = response.json()
    assert body["activity_type"] == "weather_conditions"
    assert body["details"]["fire_danger_assessment_id"] == 1
    assert body["details"]["area_name"] == "Carmel"
    assert body["details"]["fire_danger_level"] == "high"
    assert len(body["details"]["readings"]) == 1
    assert body["details"]["readings"][0]["temperature"] == 34.0


# ---------------------------------------------------------------------------
# 3. Invalid activity type -> safe client error
# ---------------------------------------------------------------------------


def test_invalid_activity_type_returns_safe_client_error():
    client = client_for(FakeOperationsActivityQueryService())

    response = client.get(endpoint("not_a_real_type", 1))

    assert 400 <= response.status_code < 500
    body_text = response.text
    for leaked in ("Traceback", "sqlalchemy", "DATABASE_URL"):
        assert leaked not in body_text


# ---------------------------------------------------------------------------
# 4. Missing entity -> 404
# ---------------------------------------------------------------------------


def test_missing_entity_returns_404_envelope():
    client = client_for(FakeOperationsActivityQueryService())

    response = client.get(endpoint("fire_event", 999))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "OPERATIONS_ACTIVITY_NOT_FOUND"


# ---------------------------------------------------------------------------
# 5. Discriminated response serializes correctly
# ---------------------------------------------------------------------------


def test_discriminator_distinguishes_variants_with_same_shaped_fields():
    fire_danger = make_fire_danger()
    fire_event = make_fire_event()
    service = FakeOperationsActivityQueryService(
        by_key={
            (OperationsActivityType.FIRE_DANGER, 1): fire_danger,
            (OperationsActivityType.FIRE_EVENT, 42): fire_event,
        }
    )
    client = client_for(service)

    fd_body = client.get(endpoint("fire_danger", 1)).json()
    fe_body = client.get(endpoint("fire_event", 42)).json()

    assert fd_body["activity_type"] != fe_body["activity_type"]
    assert set(fd_body["details"].keys()) != set(fe_body["details"].keys())


# ---------------------------------------------------------------------------
# 6. Timestamps serialize ISO-8601
# ---------------------------------------------------------------------------


def test_occurred_at_serializes_as_timezone_aware_iso8601():
    detail = make_fire_event()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_EVENT, 42): detail})
    client = client_for(service)

    body = client.get(endpoint("fire_event", 42)).json()

    value = body["occurred_at"]
    assert value.endswith("Z") or "+" in value[-6:]
    assert parse_dt(value) == ASSESSED_AT


# ---------------------------------------------------------------------------
# 7. Fire Danger INSUFFICIENT_DATA remains distinct
# ---------------------------------------------------------------------------


def test_fire_danger_insufficient_data_is_not_low():
    detail = make_fire_danger_insufficient()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_DANGER, 2): detail})
    client = client_for(service)

    body = client.get(endpoint("fire_danger", 2)).json()

    assert body["details"]["status"] == "insufficient_data"
    assert body["details"]["score"] is None
    assert body["details"]["level"] is None
    assert body["details"]["level"] != "low"


# ---------------------------------------------------------------------------
# 8. FireEvent status/confidence is persisted value
# ---------------------------------------------------------------------------


def test_fire_event_status_and_confidence_are_persisted_values():
    detail = make_fire_event()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_EVENT, 42): detail})
    client = client_for(service)

    body = client.get(endpoint("fire_event", 42)).json()

    assert body["details"]["status"] == "confirmed"
    assert body["details"]["detection_confidence"] == 0.87


# ---------------------------------------------------------------------------
# 9. GlobalPlanningRun can include >1 FireEvent
# ---------------------------------------------------------------------------


def test_global_planning_run_includes_multiple_fire_events():
    detail = make_global_planning_run()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.GLOBAL_PLANNING_RUN, 5): detail})
    client = client_for(service)

    body = client.get(endpoint("global_planning_run", 5)).json()

    assert body["details"]["fire_event_ids"] == [101, 102]
    assert body["details"]["response_plan_ids"] == [201, 202]


# ---------------------------------------------------------------------------
# 10. Response contains no secret/internal fields
# ---------------------------------------------------------------------------


def test_response_contains_no_secret_or_internal_fields():
    detail = make_global_planning_run()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.GLOBAL_PLANNING_RUN, 5): detail})
    client = client_for(service)

    body_text = client.get(endpoint("global_planning_run", 5)).text

    forbidden = (
        "_sa_instance_state",
        "DATABASE_URL",
        "postgresql://",
        "Traceback",
        "IMS_API_TOKEN",
        "FIRMS_MAP_KEY",
    )
    for field in forbidden:
        assert field not in body_text


# ---------------------------------------------------------------------------
# 11. Repeated GET has no side effects
# ---------------------------------------------------------------------------


def test_repeated_get_calls_read_method_only_and_returns_same_content():
    detail = make_fire_event()
    service = FakeOperationsActivityQueryService(by_key={(OperationsActivityType.FIRE_EVENT, 42): detail})
    client = client_for(service)

    first = client.get(endpoint("fire_event", 42)).json()
    second = client.get(endpoint("fire_event", 42)).json()

    assert first == second
    assert service.calls == [
        (OperationsActivityType.FIRE_EVENT, 42),
        (OperationsActivityType.FIRE_EVENT, 42),
    ]
    assert not hasattr(service, "save_activity")


# ---------------------------------------------------------------------------
# 12. Existing resource-detail endpoints remain compatible
# ---------------------------------------------------------------------------


def test_existing_fire_danger_and_fire_events_routes_are_still_registered():
    app = create_app()

    paths = {route.path for route in app.routes}

    assert "/api/v1/fire-danger/areas/latest" in paths
    assert "/api/v1/fire-danger/areas/{area_id}/latest" in paths
    assert "/api/v1/fire-danger/assessments/{assessment_id}" in paths
    assert "/api/v1/fire-events/active" in paths
    assert "/api/v1/fire-events/{fire_event_id}/details" in paths
    assert "/api/v1/operations/activity/{activity_type}/{entity_id}" in paths


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


def test_endpoint_is_registered_in_openapi_schema():
    app = create_app()

    schema = app.openapi()

    assert "/api/v1/operations/activity/{activity_type}/{entity_id}" in schema["paths"]
