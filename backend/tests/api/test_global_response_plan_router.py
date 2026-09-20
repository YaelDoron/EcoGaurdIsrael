"""API tests for `GET /api/v1/global-response-plan/current` (Epic 6 UI/API
Rework, Task B-BE-6).

The router under test (`src.api.routers.global_response_plan`) is not yet
registered in `src.api.routers.v1_router` (that shared-file wiring is a
follow-up left to whoever owns that file - see `test_response_plans_router.py`'s
own precedent), so these tests mount it onto a throwaway FastAPI app
themselves. FastAPI's `dependency_overrides` replaces
`GlobalResponsePlanReadService` with a fake - no real DB/service call
happens in these tests.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.global_response_plan import (
    get_global_response_plan_read_service,
    global_response_plan_router,
)
from src.api.schemas.global_response_plan import (
    GlobalEventPlan,
    GlobalOptimizationConfig,
    GlobalPlanMetrics,
    GlobalPlanResponse,
    GlobalPlanShortage,
    GlobalResponsePlanResponse,
)
from src.api.schemas.response_plans import (
    CoordinateResponse,
    ResponsePlanActionResponse,
    ResponsePlanResourceResponse,
    ResponsePlanRouteResponse,
    ResponsePlanTargetResponse,
)
from src.database.connection import DatabaseConfigurationError
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.routing import RouteStatus

STARTED_AT = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
COMPLETED_AT = STARTED_AT.replace(minute=5)
AS_OF = STARTED_AT.replace(hour=9)

ENDPOINT = "/api/v1/global-response-plan/current"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_action() -> ResponsePlanActionResponse:
    return ResponsePlanActionResponse(
        resource=ResponsePlanResourceResponse(
            resource_id="engine-1",
            station_id="station-1",
            station_name="Central Station",
            origin=CoordinateResponse(latitude=32.0, longitude=35.0),
        ),
        target=ResponsePlanTargetResponse(
            response_target_id=1, target_type="active_fire", priority_score=0.9, latitude=32.1, longitude=35.1
        ),
        route=ResponsePlanRouteResponse(
            status=RouteStatus.REACHABLE,
            eta_seconds=120.0,
            distance_meters=800.0,
            node_path=[1, 2],
            path_coordinates=None,
        ),
    )


def make_event_plan(**overrides) -> GlobalEventPlan:
    values = dict(
        fire_event_id=101,
        response_plan_id=501,
        severity_level=FireSeverityLevel.HIGH,
        severity_score=70.0,
        minimum_resources=2,
        desired_resources=4,
        assigned_resources=3,
        coverage_score=0.85,
        average_eta_seconds=140.0,
        actions=[make_action()],
        uncovered_targets=[
            ResponsePlanTargetResponse(
                response_target_id=5, target_type="active_fire", priority_score=0.3, latitude=40.0, longitude=41.0
            )
        ],
    )
    values.update(overrides)
    return GlobalEventPlan(**values)


def make_plan(**overrides) -> GlobalPlanResponse:
    values = dict(
        run_id=77,
        started_at=STARTED_AT,
        completed_at=COMPLETED_AT,
        status=GlobalPlanningRunStatus.COMPLETED,
        metrics=GlobalPlanMetrics(fitness_score=91.5, coverage_score=0.95, average_eta_seconds=180.0),
        shortage=GlobalPlanShortage(total_required=3, total_desired=6, total_assigned=4, unmet_required=0, unmet_desired=2),
        optimization_config=GlobalOptimizationConfig(
            random_seed=42, population_size=50, generation_count=100, mutation_rate=0.1, crossover_rate=0.8
        ),
        events=[make_event_plan()],
    )
    values.update(overrides)
    return GlobalPlanResponse(**values)


def make_result(**overrides) -> GlobalResponsePlanResponse:
    values = dict(as_of=AS_OF, plan=make_plan())
    values.update(overrides)
    return GlobalResponsePlanResponse(**values)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGlobalResponsePlanReadService:
    def __init__(self, result: GlobalResponsePlanResponse | None = None):
        self._result = result if result is not None else make_result()
        self.calls = 0

    def get_current(self, *, as_of=None) -> GlobalResponsePlanResponse:
        self.calls += 1
        return self._result


class RaisingGlobalResponsePlanReadService:
    def __init__(self, exc: Exception):
        self._exc = exc

    def get_current(self, *, as_of=None) -> GlobalResponsePlanResponse:
        raise self._exc


# ---------------------------------------------------------------------------
# App/client wiring
# ---------------------------------------------------------------------------


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(global_response_plan_router, prefix="/api/v1")
    return app


def client_for(service, *, raise_server_exceptions: bool = True) -> TestClient:
    app = make_app()
    app.dependency_overrides[get_global_response_plan_read_service] = lambda: service
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_current_endpoint_returns_the_full_global_plan():
    service = FakeGlobalResponsePlanReadService(make_result())
    client = client_for(service)

    response = client.get(ENDPOINT)

    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["run_id"] == 77
    assert body["plan"]["status"] == "completed"
    assert body["plan"]["metrics"]["fitness_score"] == 91.5
    assert body["plan"]["shortage"]["total_required"] == 3
    assert body["plan"]["optimization_config"]["random_seed"] == 42
    assert len(body["plan"]["events"]) == 1


def test_event_plan_carries_actions_and_uncovered_targets():
    service = FakeGlobalResponsePlanReadService(make_result())
    client = client_for(service)

    event = client.get(ENDPOINT).json()["plan"]["events"][0]

    assert event["fire_event_id"] == 101
    assert event["response_plan_id"] == 501
    assert len(event["actions"]) == 1
    assert event["actions"][0]["resource"]["resource_id"] == "engine-1"
    assert event["uncovered_targets"] == [
        {"response_target_id": 5, "target_type": "active_fire", "priority_score": 0.3, "latitude": 40.0, "longitude": 41.0}
    ]


def test_service_is_called_exactly_once():
    service = FakeGlobalResponsePlanReadService(make_result())
    client = client_for(service)

    client.get(ENDPOINT)

    assert service.calls == 1


# ---------------------------------------------------------------------------
# No materialized generation
# ---------------------------------------------------------------------------


def test_no_materialized_generation_returns_null_plan():
    service = FakeGlobalResponsePlanReadService(GlobalResponsePlanResponse(as_of=AS_OF, plan=None))
    client = client_for(service)

    response = client.get(ENDPOINT)

    assert response.status_code == 200
    assert response.json()["plan"] is None


# ---------------------------------------------------------------------------
# Optimization config absent
# ---------------------------------------------------------------------------


def test_optimization_config_null_serializes_as_null():
    service = FakeGlobalResponsePlanReadService(make_result(plan=make_plan(optimization_config=None)))
    client = client_for(service)

    body = client.get(ENDPOINT).json()

    assert body["plan"]["optimization_config"] is None


# ---------------------------------------------------------------------------
# Timestamps / enums
# ---------------------------------------------------------------------------


def test_timestamps_serialize_with_timezone_information():
    service = FakeGlobalResponsePlanReadService(make_result())
    client = client_for(service)

    body = client.get(ENDPOINT).json()

    for value in (body["as_of"], body["plan"]["started_at"], body["plan"]["completed_at"]):
        assert value.endswith("Z") or "+" in value[-6:]
        assert datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None


def test_status_and_severity_level_serialize_as_stable_lowercase_strings():
    service = FakeGlobalResponsePlanReadService(make_result())
    client = client_for(service)

    body = client.get(ENDPOINT).json()

    assert body["plan"]["status"] == "completed"
    assert body["plan"]["events"][0]["severity_level"] == "high"
    assert "GlobalPlanningRunStatus" not in str(body)
    assert "FireSeverityLevel" not in str(body)


# ---------------------------------------------------------------------------
# Safe failure
# ---------------------------------------------------------------------------


def test_service_failure_returns_5xx_without_leaking_internals():
    secret_exc = DatabaseConfigurationError("DATABASE_URL is not configured.")
    client = client_for(RaisingGlobalResponsePlanReadService(secret_exc), raise_server_exceptions=False)

    response = client.get(ENDPOINT)

    assert response.status_code >= 500
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "DatabaseConfigurationError"):
        assert leaked not in body_text


# ---------------------------------------------------------------------------
# Read-only / no forbidden imports
# ---------------------------------------------------------------------------


def test_router_module_does_not_import_forbidden_dependencies():
    forbidden_fragments = (
        "src.calculators",
        "src.agents",
        "src.external",
        "src.simulation",
        "sqlalchemy",
        "dijkstra",
        "Dijkstra",
        "genetic",
        "response_optimization",
        "planning_orchestrator",
    )
    path = Path("src/api/routers/global_response_plan.py")
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


def test_router_source_calls_no_write_or_mutation_operation():
    import inspect

    assert len(global_response_plan_router.routes) == 1
    for route in global_response_plan_router.routes:
        source = inspect.getsource(route.endpoint)
        for forbidden in (".save(", ".update(", ".delete(", ".create_", "mutate", "optimize"):
            assert forbidden not in source


# ---------------------------------------------------------------------------
# Not yet wired into __init__.py (guardrail check, mirrors response_plans)
# ---------------------------------------------------------------------------


def test_router_file_does_not_touch_shared_integration_files():
    """Guardrail smoke-check: this router's own module must not import from
    or modify `src.api.routers.__init__`/`src.api.dependencies` - it defines
    its own local dependency factory instead (see module docstring)."""
    path = Path("src/api/routers/global_response_plan.py")
    source = path.read_text(encoding="utf-8")
    assert "src.api.dependencies" not in source
