"""API tests for the Operations Overview endpoint (Task A6, Part 32).

Uses FastAPI's dependency_overrides to replace OperationsOverviewQueryService
with a fake returning controlled read-model data - no real repository/DB/
agent/calculator call happens in these tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_operations_overview_query_service
from src.models.active_fire_events import ActiveFireEventSeveritySummary, ActiveFireEventSummary
from src.models.fire_danger_areas import FireDangerAreaAssessmentSummary, FireDangerAreaSnapshot
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.operations_activity import OperationsActivityLocation, OperationsActivityType
from src.models.operations_overview import (
    FireDangerActivityPreview,
    FireEventActivityPreview,
    FireSeverityActivityPreview,
    GlobalPlanningRunActivityPreview,
    NewsReportActivityPreview,
    OperationsActivityFeed,
    OperationsActivityFeedItem,
    OperationsOverviewSnapshot,
    OperationsSimulationSummary,
    SatelliteHotspotActivityPreview,
)
from src.services.simulation_control.simulation_run_manager import (
    SimulationCurrentEventSnapshot,
    SimulationRunSnapshot,
    SimulationRunState,
)

GENERATED_AT = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
OVERVIEW_ENDPOINT = "/api/v1/operations/overview"


class FakeOperationsOverviewQueryService:
    def __init__(self, snapshot: OperationsOverviewSnapshot):
        self._snapshot = snapshot
        self.calls: list[int] = []

    def get_overview(self, *, activity_limit: int = 30, as_of=None):
        self.calls.append(activity_limit)
        return self._snapshot


def client_for(snapshot: OperationsOverviewSnapshot, *, raise_server_exceptions: bool = True) -> TestClient:
    app = create_app()
    service = FakeOperationsOverviewQueryService(snapshot)
    app.dependency_overrides[get_operations_overview_query_service] = lambda: service
    client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
    client.fake_service = service  # type: ignore[attr-defined]
    return client


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def empty_snapshot(*, generated_at: datetime = GENERATED_AT) -> OperationsOverviewSnapshot:
    return OperationsOverviewSnapshot(
        generated_at=generated_at,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(),
        active_fires=(),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )


def make_fire_danger_area(**overrides) -> FireDangerAreaSnapshot:
    values = dict(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessment=FireDangerAreaAssessmentSummary(
            assessment_id=1,
            status=FireDangerAssessmentStatus.VALID,
            score=42.5,
            level=FireDangerLevel.VERY_HIGH,
            assessed_at=GENERATED_AT,
            methodology="FOSBERG_FFWI",
            methodology_version="1.0",
        ),
    )
    values.update(overrides)
    return FireDangerAreaSnapshot(**values)


def make_active_fire(**overrides) -> ActiveFireEventSummary:
    values = dict(
        fire_event_id=42,
        status=FireEventStatus.CONFIRMED,
        latitude=32.7,
        longitude=35.0,
        detection_confidence=0.84,
        detected_at=GENERATED_AT,
        updated_at=GENERATED_AT,
        created_at=GENERATED_AT,
        severity=ActiveFireEventSeveritySummary(
            assessment_id=99,
            status=FireSeverityAssessmentStatus.VALID,
            score=57.3,
            level=FireSeverityLevel.HIGH,
            assessed_at=GENERATED_AT,
        ),
    )
    values.update(overrides)
    return ActiveFireEventSummary(**values)


def make_feed_item(**overrides) -> OperationsActivityFeedItem:
    values = dict(
        activity_id="fire_danger:1",
        activity_type=OperationsActivityType.FIRE_DANGER,
        entity_id=1,
        occurred_at=GENERATED_AT,
        available_at=GENERATED_AT,
        title="Fire Danger Assessment - Carmel",
        location=OperationsActivityLocation(latitude=32.731, longitude=35.046),
        preview=FireDangerActivityPreview(
            area_name="Carmel", status=FireDangerAssessmentStatus.VALID, level=FireDangerLevel.VERY_HIGH, score=42.5
        ),
    )
    values.update(overrides)
    return OperationsActivityFeedItem(**values)


def make_simulation_snapshot(**overrides) -> SimulationRunSnapshot:
    values = dict(
        run_id="run-1",
        state=SimulationRunState.RUNNING,
        preset_id="operations_demo",
        seed=42,
        mode="automatic",
        simulation_duration_seconds=120,
        events_total=18,
        events_completed=5,
        events_succeeded=5,
        events_failed=0,
        current_event=SimulationCurrentEventSnapshot(
            event_index=5, incident_id="incident-1", event_type="weather", timestamp_offset_sec=60
        ),
        started_at=GENERATED_AT,
        completed_at=None,
        wall_clock_elapsed_seconds=12.5,
        last_message="Simulation running.",
        error=None,
    )
    values.update(overrides)
    return SimulationRunSnapshot(**values)


# ---------------------------------------------------------------------------
# 1. HTTP 200 empty state
# ---------------------------------------------------------------------------


def test_empty_state_returns_200():
    client = client_for(empty_snapshot())

    response = client.get(OVERVIEW_ENDPOINT)

    assert response.status_code == 200
    body = response.json()
    assert body["fire_danger_areas"] == []
    assert body["active_fires"] == []
    assert body["activity_feed"]["items"] == []
    assert body["simulation"] == {"enabled": False, "run": None}


# ---------------------------------------------------------------------------
# 2/3/4. activity_limit handling
# ---------------------------------------------------------------------------


def test_default_activity_limit_is_30():
    client = client_for(empty_snapshot())

    client.get(OVERVIEW_ENDPOINT)

    assert client.fake_service.calls == [30]


def test_custom_valid_activity_limit_is_forwarded():
    client = client_for(empty_snapshot())

    client.get(OVERVIEW_ENDPOINT, params={"activity_limit": 10})

    assert client.fake_service.calls == [10]


def test_invalid_activity_limit_is_rejected():
    client = client_for(empty_snapshot())

    too_low = client.get(OVERVIEW_ENDPOINT, params={"activity_limit": 0})
    too_high = client.get(OVERVIEW_ENDPOINT, params={"activity_limit": 101})

    assert too_low.status_code == 422
    assert too_high.status_code == 422


# ---------------------------------------------------------------------------
# 5. Fire Danger schema
# ---------------------------------------------------------------------------


def test_fire_danger_areas_schema():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(make_fire_danger_area(),),
        active_fires=(),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    area = body["fire_danger_areas"][0]
    assert area["area_name"] == "Carmel"
    assert area["center"] == {"latitude": 32.731, "longitude": 35.046}
    assert area["assessment"]["level"] == "very_high"
    assert area["assessment"]["score"] == 42.5


# ---------------------------------------------------------------------------
# 6/7. Active-fire + latest-Severity schema
# ---------------------------------------------------------------------------


def test_active_fire_and_severity_schema():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(),
        active_fires=(make_active_fire(),),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    fire = body["active_fires"][0]
    assert fire["fire_event_id"] == 42
    assert fire["status"] == "confirmed"
    assert fire["detection_confidence"] == 0.84
    assert fire["severity"]["assessment_id"] == 99
    assert fire["severity"]["level"] == "high"
    assert fire["severity"]["score"] == 57.3


def test_active_fire_with_null_severity():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(),
        active_fires=(make_active_fire(severity=None),),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    assert body["active_fires"][0]["severity"] is None


# ---------------------------------------------------------------------------
# 8. Activity-feed schema
# ---------------------------------------------------------------------------


def test_activity_feed_schema_discriminates_by_type():
    news_item = make_feed_item(
        activity_id="news_report:3",
        activity_type=OperationsActivityType.NEWS_REPORT,
        entity_id=3,
        title="Wildfire reported",
        location=None,
        preview=NewsReportActivityPreview(source="Example Feed", headline="Wildfire reported"),
    )
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(),
        active_fires=(),
        activity_feed=OperationsActivityFeed(items=(make_feed_item(), news_item), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    items = body["activity_feed"]["items"]
    assert body["activity_feed"]["limit"] == 30
    assert items[0]["activity_type"] == "fire_danger"
    assert items[0]["preview"]["area_name"] == "Carmel"
    assert items[1]["activity_type"] == "news_report"
    assert items[1]["preview"]["headline"] == "Wildfire reported"
    assert items[1]["activity_id"] == "news_report:3"


def test_all_six_preview_types_serialize():
    items = (
        make_feed_item(),
        make_feed_item(
            activity_id="satellite_hotspot:7", activity_type=OperationsActivityType.SATELLITE_HOTSPOT,
            entity_id=7, title="Satellite Hotspot", location=OperationsActivityLocation(latitude=32.7, longitude=35.0),
            preview=SatelliteHotspotActivityPreview(confidence="h", frp=45.6),
        ),
        make_feed_item(
            activity_id="fire_event:42", activity_type=OperationsActivityType.FIRE_EVENT,
            entity_id=42, title="Fire Event #42", location=OperationsActivityLocation(latitude=32.7, longitude=35.0),
            preview=FireEventActivityPreview(status=FireEventStatus.CONFIRMED, confidence=0.84),
        ),
        make_feed_item(
            activity_id="fire_severity:99", activity_type=OperationsActivityType.FIRE_SEVERITY,
            entity_id=99, title="Fire Severity Assessment", location=None,
            preview=FireSeverityActivityPreview(fire_event_id=42, level=FireSeverityLevel.HIGH, score=57.3),
        ),
        make_feed_item(
            activity_id="global_planning_run:5", activity_type=OperationsActivityType.GLOBAL_PLANNING_RUN,
            entity_id=5, title="Global Response Plan", location=None,
            preview=GlobalPlanningRunActivityPreview(status=GlobalPlanningRunStatus.COMPLETED, fire_event_count=2),
        ),
    )
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(),
        active_fires=(),
        activity_feed=OperationsActivityFeed(items=items, limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    assert len(body["activity_feed"]["items"]) == 5
    assert body["activity_feed"]["items"][4]["preview"]["fire_event_count"] == 2


# ---------------------------------------------------------------------------
# 9. Simulation schema
# ---------------------------------------------------------------------------


def test_simulation_schema_enabled_with_run():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=True, run=make_simulation_snapshot()),
        fire_danger_areas=(),
        active_fires=(),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    assert body["simulation"]["enabled"] is True
    assert body["simulation"]["run"]["state"] == "running"
    assert body["simulation"]["run"]["events_completed"] == 5
    assert body["simulation"]["run"]["current_event"]["incident_id"] == "incident-1"


def test_simulation_schema_enabled_no_run():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=True, run=None),
        fire_danger_areas=(),
        active_fires=(),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    assert body["simulation"] == {"enabled": True, "run": None}


# ---------------------------------------------------------------------------
# 10. Timezone-aware ISO timestamps
# ---------------------------------------------------------------------------


def test_generated_at_is_timezone_aware_iso8601():
    client = client_for(empty_snapshot())

    body = client.get(OVERVIEW_ENDPOINT).json()

    value = body["generated_at"]
    assert value.endswith("Z") or "+" in value[-6:]
    assert parse_dt(value) == GENERATED_AT


# ---------------------------------------------------------------------------
# 11/13. No ORM/internal or secret fields
# ---------------------------------------------------------------------------


def test_response_contains_no_orm_or_secret_fields():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=True, run=make_simulation_snapshot()),
        fire_danger_areas=(make_fire_danger_area(),),
        active_fires=(make_active_fire(),),
        activity_feed=OperationsActivityFeed(items=(make_feed_item(),), limit=30),
    )
    client = client_for(snapshot)

    body_text = client.get(OVERVIEW_ENDPOINT).text

    forbidden = (
        "_sa_instance_state",
        "DATABASE_URL",
        "postgresql://",
        "Traceback",
        "IMS_API_TOKEN",
        "FIRMS_MAP_KEY",
        "danger_level",
    )
    for field in forbidden:
        assert field not in body_text


def test_top_level_response_has_only_defined_sections():
    client = client_for(empty_snapshot())

    body = client.get(OVERVIEW_ENDPOINT).json()

    assert set(body.keys()) == {"generated_at", "simulation", "fire_danger_areas", "active_fires", "activity_feed"}


# ---------------------------------------------------------------------------
# 12. No FireEvent area_name
# ---------------------------------------------------------------------------


def test_active_fire_response_has_no_area_name_field():
    snapshot = OperationsOverviewSnapshot(
        generated_at=GENERATED_AT,
        simulation=OperationsSimulationSummary(enabled=False, run=None),
        fire_danger_areas=(),
        active_fires=(make_active_fire(),),
        activity_feed=OperationsActivityFeed(items=(), limit=30),
    )
    client = client_for(snapshot)

    body = client.get(OVERVIEW_ENDPOINT).json()

    assert "area_name" not in body["active_fires"][0]


# ---------------------------------------------------------------------------
# 14. Repeated GET causes no writes
# ---------------------------------------------------------------------------


def test_repeated_get_calls_read_method_only():
    snapshot = empty_snapshot()
    client = client_for(snapshot)

    client.get(OVERVIEW_ENDPOINT)
    client.get(OVERVIEW_ENDPOINT)

    assert client.fake_service.calls == [30, 30]
    assert not hasattr(client.fake_service, "save_overview")


def test_polling_get_many_times_never_triggers_a_reset(monkeypatch):
    """Task 5, Part 2/23: even many rapid GETs (simulating a live-polling
    browser tab) must never call reset_demo_state - a browser refresh/poll
    must never destroy an active run's data."""
    from src.simulation.demo_state_reset_service import DemoStateResetService

    reset_calls = []
    monkeypatch.setattr(
        DemoStateResetService, "reset_demo_state", lambda self: reset_calls.append(1)
    )
    client = client_for(empty_snapshot())

    for _ in range(20):
        response = client.get(OVERVIEW_ENDPOINT)
        assert response.status_code == 200

    assert reset_calls == []
    assert client.fake_service.calls == [30] * 20


def test_router_source_calls_no_write_operation():
    import inspect

    from src.api.routers.operations_overview import operations_overview_router

    for route in operations_overview_router.routes:
        source = inspect.getsource(route.endpoint)
        for forbidden in (
            ".save_",
            ".create_",
            ".record_",
            "Agent(",
            "Calculator(",
            "reset_demo_state",
            "DemoStateResetService",
        ):
            assert forbidden not in source


def test_router_module_never_imports_the_demo_reset_service():
    """Task 5, Part 2/23: GET /operations/overview must remain read-only -
    reset is triggered ONLY by POST /simulation/runs with reset_demo_state=true
    (src/api/routers/simulation.py). This is a static guard against ever
    wiring reset into a read path, not just a runtime behavior check.

    Paths are resolved relative to this test file (not the CWD) - unlike
    the pre-existing similarly-shaped guards in test_response_plans_router.py/
    test_response_plan_presenter.py, which break when pytest is invoked from
    inside backend/ rather than the repo root.
    """
    import ast
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[2]
    for path in (
        backend_root / "src/api/routers/operations_overview.py",
        backend_root / "src/services/operations/operations_overview_query_service.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            assert "demo_state_reset_service" not in module, f"{path} imports the reset service: {module!r}"
            assert "DemoStateResetService" not in module


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


def test_endpoint_is_registered_in_openapi_schema():
    app = create_app()

    schema = app.openapi()

    assert "/api/v1/operations/overview" in schema["paths"]
