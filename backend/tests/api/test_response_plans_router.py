"""API tests for the US 6.3 response-plan endpoints (Tasks 2-3-4-5):

* `GET /api/v1/fire-events/{fire_event_id}/response-plan` (Task 2)
* `GET /api/v1/response-plans/{plan_id}` (Task 3)

Both now return the fully enriched `ResponsePlanDetailResponse` (Task 4)
produced by `ResponsePlanPresenter`, wired in by Task 5:

    ResponsePlanDetailsService -> ResponsePlanDetails -> ResponsePlanPresenter -> HTTP

The router under test (`src.api.routers.response_plans`) is not yet
registered in `src.api.routers.v1_router` (that shared-file wiring is a
follow-up left to whoever owns that file), so these tests mount it onto a
throwaway FastAPI app under `/api/v1` themselves. FastAPI's
`dependency_overrides` replaces both `ResponsePlanDetailsService` and
`ResponsePlanPresenter` with fakes - no real DB/service/repository call
happens in these tests.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.response_plans import (
    get_response_eligibility_reader,
    get_response_plan_details_service,
    get_response_plan_presenter,
    response_plans_router,
)
from src.api.schemas.response_plans import (
    CoordinateResponse,
    ResponsePlanActionResponse,
    ResponsePlanDetailResponse,
    ResponsePlanMetricsResponse,
    ResponsePlanResourceResponse,
    ResponsePlanRouteResponse,
    ResponsePlanTargetResponse,
)
from src.database.connection import DatabaseConfigurationError
from src.models.response_plan_details import ResponsePlanDetails
from src.models.response_plan_status import ResponsePlanStatus
from src.models.routing import RouteStatus

GENERATED_AT = datetime(2026, 9, 17, 13, 20, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Builders for the enriched response DTO returned by the (faked) presenter.
# ---------------------------------------------------------------------------


def make_action_response(**overrides) -> ResponsePlanActionResponse:
    values = dict(
        resource=ResponsePlanResourceResponse(
            resource_id="engine-1",
            station_id="station-1",
            station_name="Central Station",
            origin=CoordinateResponse(latitude=32.0, longitude=35.0),
        ),
        target=ResponsePlanTargetResponse(
            response_target_id=1,
            target_type="active_fire",
            priority_score=0.75,
            latitude=32.1,
            longitude=35.1,
        ),
        route=ResponsePlanRouteResponse(
            status=RouteStatus.REACHABLE,
            eta_seconds=120.0,
            distance_meters=800.0,
            node_path=[1, 2, 3],
            path_coordinates=[
                CoordinateResponse(latitude=1.0, longitude=1.0),
                CoordinateResponse(latitude=2.0, longitude=2.0),
                CoordinateResponse(latitude=3.0, longitude=3.0),
            ],
        ),
    )
    values.update(overrides)
    return ResponsePlanActionResponse(**values)


def make_plan_response(**overrides) -> ResponsePlanDetailResponse:
    values = dict(
        plan_id=7,
        fire_event_id=3,
        response_target_set_id=9,
        route_planning_run_id=11,
        generated_at=GENERATED_AT,
        methodology="genetic_algorithm",
        methodology_version="1.0.0",
        random_seed=42,
        status=ResponsePlanStatus.COMPLETE,
        is_current=True,
        metrics=ResponsePlanMetricsResponse(plan_score=95.5, coverage_score=0.9, average_eta_seconds=150.0),
        actions=[make_action_response()],
        uncovered_targets=[
            ResponsePlanTargetResponse(
                response_target_id=5, target_type="active_fire", priority_score=0.3, latitude=40.0, longitude=41.0
            )
        ],
        baseline_comparison=None,
        optimization_config=None,
        no_resources_during_planning=False,
    )
    values.update(overrides)
    return ResponsePlanDetailResponse(**values)


def make_details(**overrides) -> ResponsePlanDetails:
    """A minimal ResponsePlanDetails to pass through the fake service - its
    field values are irrelevant to the router, since the fake presenter
    below returns a pre-built ResponsePlanDetailResponse regardless."""
    values = dict(
        plan_id=7,
        fire_event_id=3,
        response_target_set_id=9,
        route_planning_run_id=11,
        generated_at=GENERATED_AT,
        methodology="genetic_algorithm",
        methodology_version="1.0.0",
        random_seed=42,
        is_current=True,
        plan_score=95.5,
        coverage_score=0.9,
        average_eta_seconds=150.0,
        actions=(),
        uncovered_target_ids=(),
        baseline_comparison=None,
        optimization_config=None,
    )
    values.update(overrides)
    return ResponsePlanDetails(**values)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResponsePlanDetailsService:
    def __init__(self, details: ResponsePlanDetails | None = None, *, by_id: ResponsePlanDetails | None = None):
        self._details = details
        self._by_id_details = by_id
        self.calls: list[int] = []
        self.by_id_calls: list[int] = []

    def get_current_plan_details(self, fire_event_id: int) -> ResponsePlanDetails | None:
        self.calls.append(fire_event_id)
        return self._details

    def get_plan_details_by_id(self, plan_id: int) -> ResponsePlanDetails | None:
        self.by_id_calls.append(plan_id)
        return self._by_id_details


class RaisingResponsePlanDetailsService:
    def __init__(self, exc: Exception):
        self._exc = exc

    def get_current_plan_details(self, fire_event_id: int) -> ResponsePlanDetails | None:
        raise self._exc

    def get_plan_details_by_id(self, plan_id: int) -> ResponsePlanDetails | None:
        raise self._exc


class FakeResponsePlanPresenter:
    def __init__(self, response: ResponsePlanDetailResponse | None = None):
        self._response = response if response is not None else make_plan_response()
        self.calls: list[ResponsePlanDetails] = []

    def present(self, details: ResponsePlanDetails) -> ResponsePlanDetailResponse:
        self.calls.append(details)
        return self._response


class RaisingResponsePlanPresenter:
    def __init__(self, exc: Exception):
        self._exc = exc

    def present(self, details: ResponsePlanDetails) -> ResponsePlanDetailResponse:
        raise self._exc


# ---------------------------------------------------------------------------
# App/client wiring
# ---------------------------------------------------------------------------


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(response_plans_router, prefix="/api/v1")
    return app


class FakeEligibilityReader:
    def __init__(self, eligible_ids=()):
        self.eligible_ids = set(eligible_ids)

    def is_response_eligible(self, fire_event_id):
        return fire_event_id in self.eligible_ids


def client_for(service, presenter, *, raise_server_exceptions: bool = True, eligible_ids=()) -> TestClient:
    app = make_app()
    app.dependency_overrides[get_response_plan_details_service] = lambda: service
    app.dependency_overrides[get_response_plan_presenter] = lambda: presenter
    app.dependency_overrides[get_response_eligibility_reader] = lambda: FakeEligibilityReader(eligible_ids)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def endpoint(fire_event_id) -> str:
    return f"/api/v1/fire-events/{fire_event_id}/response-plan"


def plan_by_id_endpoint(plan_id) -> str:
    return f"/api/v1/response-plans/{plan_id}"


# ---------------------------------------------------------------------------
# 1: current-plan endpoint returns the enriched plan
# ---------------------------------------------------------------------------


def test_current_plan_endpoint_returns_enriched_plan():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    response = client.get(endpoint(3))

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"plan", "plan_status"}
    assert body["plan_status"] == "available"
    assert body["plan"]["plan_id"] == 7


# ---------------------------------------------------------------------------
# 2: plan-by-id endpoint returns the enriched plan
# ---------------------------------------------------------------------------


def test_plan_by_id_endpoint_returns_enriched_plan():
    service = FakeResponsePlanDetailsService(by_id=make_details(plan_id=15, fire_event_id=4))
    presenter = FakeResponsePlanPresenter(make_plan_response(plan_id=15, fire_event_id=4))
    client = client_for(service, presenter)

    response = client.get(plan_by_id_endpoint(15))

    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["plan_id"] == 15
    assert body["plan"]["fire_event_id"] == 4


# ---------------------------------------------------------------------------
# 3: persisted plan status appears correctly
# ---------------------------------------------------------------------------


def test_persisted_plan_status_appears_in_response():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response(status=ResponsePlanStatus.PARTIAL))
    client = client_for(service, presenter)

    body = client.get(endpoint(3)).json()

    assert body["plan"]["status"] == "partial"


# ---------------------------------------------------------------------------
# 4-5: is_current preserved for current vs superseded
# ---------------------------------------------------------------------------


def test_is_current_true_is_preserved_for_current_plan():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response(is_current=True))
    client = client_for(service, presenter)

    body = client.get(endpoint(3)).json()

    assert body["plan"]["is_current"] is True


def test_is_current_false_is_preserved_for_superseded_plan():
    service = FakeResponsePlanDetailsService(by_id=make_details(is_current=False))
    presenter = FakeResponsePlanPresenter(make_plan_response(is_current=False, status=ResponsePlanStatus.COMPLETE))
    client = client_for(service, presenter)

    body = client.get(plan_by_id_endpoint(7)).json()

    assert body["plan"]["is_current"] is False


# ---------------------------------------------------------------------------
# 6-9: action resource/target/route enrichment passes through unchanged
# ---------------------------------------------------------------------------


def test_action_resource_and_station_data_appears_correctly():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    resource = client.get(endpoint(3)).json()["plan"]["actions"][0]["resource"]

    assert resource == {
        "resource_id": "engine-1",
        "station_id": "station-1",
        "station_name": "Central Station",
        "origin": {"latitude": 32.0, "longitude": 35.0},
    }


def test_action_target_coordinates_type_and_priority_appear_correctly():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    target = client.get(endpoint(3)).json()["plan"]["actions"][0]["target"]

    assert target == {
        "response_target_id": 1,
        "target_type": "active_fire",
        "priority_score": 0.75,
        "latitude": 32.1,
        "longitude": 35.1,
    }


def test_persisted_eta_and_distance_appear_unchanged():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    route = client.get(endpoint(3)).json()["plan"]["actions"][0]["route"]

    assert route["eta_seconds"] == 120.0
    assert route["distance_meters"] == 800.0
    assert route["status"] == "reachable"


def test_node_path_and_path_coordinates_appear_correctly():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    route = client.get(endpoint(3)).json()["plan"]["actions"][0]["route"]

    assert route["node_path"] == [1, 2, 3]
    assert route["path_coordinates"] == [
        {"latitude": 1.0, "longitude": 1.0},
        {"latitude": 2.0, "longitude": 2.0},
        {"latitude": 3.0, "longitude": 3.0},
    ]


# ---------------------------------------------------------------------------
# 10: uncovered targets returned as full DTOs
# ---------------------------------------------------------------------------


def test_uncovered_targets_are_returned_as_full_target_dtos():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    uncovered = client.get(endpoint(3)).json()["plan"]["uncovered_targets"]

    assert uncovered == [
        {
            "response_target_id": 5,
            "target_type": "active_fire",
            "priority_score": 0.3,
            "latitude": 40.0,
            "longitude": 41.0,
        }
    ]


# ---------------------------------------------------------------------------
# 11: no_resources_during_planning
# ---------------------------------------------------------------------------


def test_no_resources_during_planning_is_returned_correctly():
    service = FakeResponsePlanDetailsService(make_details())
    presenter = FakeResponsePlanPresenter(make_plan_response(no_resources_during_planning=True))
    client = client_for(service, presenter)

    body = client.get(endpoint(3)).json()

    assert body["plan"]["no_resources_during_planning"] is True


# ---------------------------------------------------------------------------
# 12: baseline/config remain unchanged
# ---------------------------------------------------------------------------


def test_baseline_comparison_and_optimization_config_are_preserved():
    from src.api.schemas.response_plans import BaselineComparisonResponse, OptimizationConfigResponse

    presenter = FakeResponsePlanPresenter(
        make_plan_response(
            baseline_comparison=BaselineComparisonResponse(
                baseline_score=50.0,
                baseline_coverage_score=0.5,
                baseline_average_eta_seconds=300.0,
                score_difference=10.0,
                improvement_percentage=20.0,
            ),
            optimization_config=OptimizationConfigResponse(
                population_size=50,
                generation_count=100,
                mutation_rate=0.1,
                crossover_rate=0.8,
                eta_reference_seconds=600.0,
                initial_assignment_probability=0.5,
                tournament_size=3,
                elitism_count=2,
            ),
        )
    )
    service = FakeResponsePlanDetailsService(make_details())
    client = client_for(service, presenter)

    body = client.get(endpoint(3)).json()["plan"]

    assert body["baseline_comparison"] == {
        "baseline_score": 50.0,
        "baseline_coverage_score": 0.5,
        "baseline_average_eta_seconds": 300.0,
        "score_difference": 10.0,
        "improvement_percentage": 20.0,
    }
    assert body["optimization_config"]["population_size"] == 50
    assert body["optimization_config"]["elitism_count"] == 2


# ---------------------------------------------------------------------------
# 13-14: no-plan / nonexistent-plan cases
# ---------------------------------------------------------------------------


def test_current_endpoint_with_no_plan_returns_null_plan():
    service = FakeResponsePlanDetailsService(None)
    presenter = FakeResponsePlanPresenter()
    client = client_for(service, presenter)

    response = client.get(endpoint(3))

    assert response.status_code == 200
    assert response.json() == {"plan": None, "plan_status": "not_applicable"}


def test_confirmed_event_without_a_plan_yet_is_reported_as_generating():
    client = client_for(FakeResponsePlanDetailsService(None), FakeResponsePlanPresenter(), eligible_ids=[3])

    response = client.get(endpoint(3))

    assert response.status_code == 200
    assert response.json() == {"plan": None, "plan_status": "generating"}


def test_plan_by_id_missing_plan_returns_safe_404():
    service = FakeResponsePlanDetailsService(by_id=None)
    presenter = FakeResponsePlanPresenter()
    client = client_for(service, presenter)

    response = client.get(plan_by_id_endpoint(9999))

    assert response.status_code == 404
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "repository", "Repository"):
        assert leaked not in body_text


def test_plan_by_id_missing_plan_body_matches_shared_error_shape():
    service = FakeResponsePlanDetailsService(by_id=None)
    presenter = FakeResponsePlanPresenter()
    client = client_for(service, presenter)

    response = client.get(plan_by_id_endpoint(9999))

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "RESPONSE_PLAN_NOT_FOUND",
            "message": "The requested response plan was not found.",
        }
    }
    assert "detail" not in response.json()


# ---------------------------------------------------------------------------
# 15-16: presenter call count
# ---------------------------------------------------------------------------


def test_presenter_is_called_exactly_once_for_current_plan():
    details = make_details()
    service = FakeResponsePlanDetailsService(details)
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    client.get(endpoint(3))

    assert presenter.calls == [details]


def test_presenter_is_called_exactly_once_for_plan_by_id():
    details = make_details(plan_id=7)
    service = FakeResponsePlanDetailsService(by_id=details)
    presenter = FakeResponsePlanPresenter(make_plan_response())
    client = client_for(service, presenter)

    client.get(plan_by_id_endpoint(7))

    assert presenter.calls == [details]


def test_presenter_is_not_called_when_no_current_plan_exists():
    service = FakeResponsePlanDetailsService(None)
    presenter = FakeResponsePlanPresenter()
    client = client_for(service, presenter)

    client.get(endpoint(3))

    assert presenter.calls == []


def test_presenter_is_not_called_when_plan_by_id_does_not_exist():
    service = FakeResponsePlanDetailsService(by_id=None)
    presenter = FakeResponsePlanPresenter()
    client = client_for(service, presenter)

    client.get(plan_by_id_endpoint(9999))

    assert presenter.calls == []


# ---------------------------------------------------------------------------
# Invalid input (unchanged from Tasks 2/3)
# ---------------------------------------------------------------------------


def test_zero_fire_event_id_is_rejected():
    client = client_for(FakeResponsePlanDetailsService(make_details()), FakeResponsePlanPresenter())

    response = client.get(endpoint(0))

    assert response.status_code == 422


def test_negative_fire_event_id_is_rejected():
    client = client_for(FakeResponsePlanDetailsService(make_details()), FakeResponsePlanPresenter())

    response = client.get(endpoint(-5))

    assert response.status_code == 422


def test_non_integer_fire_event_id_is_rejected():
    client = client_for(FakeResponsePlanDetailsService(make_details()), FakeResponsePlanPresenter())

    response = client.get(endpoint("abc"))

    assert response.status_code == 422


def test_plan_by_id_zero_is_rejected():
    client = client_for(FakeResponsePlanDetailsService(by_id=make_details()), FakeResponsePlanPresenter())

    response = client.get(plan_by_id_endpoint(0))

    assert response.status_code == 422


def test_plan_by_id_negative_is_rejected():
    client = client_for(FakeResponsePlanDetailsService(by_id=make_details()), FakeResponsePlanPresenter())

    response = client.get(plan_by_id_endpoint(-1))

    assert response.status_code == 422


def test_plan_by_id_non_integer_is_rejected():
    client = client_for(FakeResponsePlanDetailsService(by_id=make_details()), FakeResponsePlanPresenter())

    response = client.get(plan_by_id_endpoint("xyz"))

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Safe failure - service and presenter
# ---------------------------------------------------------------------------


def test_service_failure_returns_5xx_without_leaking_internals():
    secret_exc = DatabaseConfigurationError("DATABASE_URL is not configured.")
    client = client_for(
        RaisingResponsePlanDetailsService(secret_exc), FakeResponsePlanPresenter(), raise_server_exceptions=False
    )

    response = client.get(endpoint(3))

    assert response.status_code >= 500
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "DatabaseConfigurationError"):
        assert leaked not in body_text


def test_plan_by_id_service_failure_returns_5xx_without_leaking_internals():
    secret_exc = DatabaseConfigurationError("DATABASE_URL is not configured.")
    client = client_for(
        RaisingResponsePlanDetailsService(secret_exc), FakeResponsePlanPresenter(), raise_server_exceptions=False
    )

    response = client.get(plan_by_id_endpoint(7))

    assert response.status_code >= 500
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "DatabaseConfigurationError"):
        assert leaked not in body_text


def test_presenter_failure_returns_5xx_without_leaking_internals():
    secret_exc = DatabaseConfigurationError("DATABASE_URL is not configured.")
    client = client_for(
        FakeResponsePlanDetailsService(make_details()),
        RaisingResponsePlanPresenter(secret_exc),
        raise_server_exceptions=False,
    )

    response = client.get(endpoint(3))

    assert response.status_code >= 500
    body_text = response.text
    for leaked in ("DATABASE_URL", "postgresql", "Traceback", "sqlalchemy", "DatabaseConfigurationError"):
        assert leaked not in body_text


# ---------------------------------------------------------------------------
# 17: read-only / no forbidden imports
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
        "response_planning_refresh",
        "planning_orchestrator",
    )
    path = (Path(__file__).resolve().parents[2] / "src/api/routers/response_plans.py")
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

    assert len(response_plans_router.routes) == 2
    for route in response_plans_router.routes:
        source = inspect.getsource(route.endpoint)
        for forbidden in (".save(", ".update(", ".delete(", ".create_", "mutate", "refresh_plan", "optimize"):
            assert forbidden not in source
