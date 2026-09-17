"""Tests for the US 6.3 final API schema (`src.api.schemas.response_plans`).

These DTOs are pure data containers - the Task 1-3 flat contract and its
`to_response_plan_response` mapper have been retired (Task 5) now that both
endpoints return the enriched `ResponsePlanDetailResponse` via
`ResponsePlanPresenter`. This file therefore tests the DTOs' own
construction/serialization behavior only (enum stability, Optional
None-handling, envelope null-plan case) - the read-only enrichment logic
that populates them from persisted data lives in `ResponsePlanPresenter`
and is tested in `tests/api/test_response_plan_presenter.py`.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

from src.api.schemas.response_plans import (
    BaselineComparisonResponse,
    CoordinateResponse,
    OptimizationConfigResponse,
    ResponsePlanActionResponse,
    ResponsePlanDetailResponse,
    ResponsePlanEnvelopeResponse,
    ResponsePlanMetricsResponse,
    ResponsePlanResourceResponse,
    ResponsePlanRouteResponse,
    ResponsePlanTargetResponse,
)
from src.models.response_plan_status import ResponsePlanStatus
from src.models.routing import RouteStatus

GENERATED_AT = datetime(2026, 9, 17, 13, 20, tzinfo=timezone.utc)


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
            path_coordinates=[CoordinateResponse(latitude=1.0, longitude=1.0)],
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
        uncovered_targets=[],
        baseline_comparison=None,
        optimization_config=None,
        no_resources_during_planning=False,
    )
    values.update(overrides)
    return ResponsePlanDetailResponse(**values)


def test_plan_status_enum_serializes_to_stable_lowercase_value():
    response = make_plan_response(status=ResponsePlanStatus.PARTIAL)

    assert response.model_dump(mode="json")["status"] == "partial"


def test_route_status_enum_serializes_to_stable_lowercase_value():
    response = make_plan_response()

    assert response.model_dump(mode="json")["actions"][0]["route"]["status"] == "reachable"


def test_route_status_none_is_allowed_when_no_route_result_found():
    action = make_action_response(
        route=ResponsePlanRouteResponse(
            status=None, eta_seconds=None, distance_meters=None, node_path=None, path_coordinates=None
        )
    )

    response = make_plan_response(actions=[action])

    assert response.actions[0].route.status is None


def test_target_optional_fields_accept_none():
    target = ResponsePlanTargetResponse(
        response_target_id=99, target_type=None, priority_score=None, latitude=None, longitude=None
    )

    assert target.response_target_id == 99
    assert target.target_type is None
    assert target.priority_score is None
    assert target.latitude is None
    assert target.longitude is None


def test_resource_optional_fields_accept_none():
    resource = ResponsePlanResourceResponse(
        resource_id="engine-1", station_id="station-1", station_name=None, origin=None
    )

    assert resource.station_name is None
    assert resource.origin is None


def test_baseline_comparison_optional_fields_accept_none():
    comparison = BaselineComparisonResponse(
        baseline_score=50.0,
        baseline_coverage_score=0.5,
        baseline_average_eta_seconds=None,
        score_difference=10.0,
        improvement_percentage=None,
    )

    assert comparison.baseline_average_eta_seconds is None
    assert comparison.improvement_percentage is None


def test_optimization_config_fields_round_trip():
    config = OptimizationConfigResponse(
        population_size=50,
        generation_count=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        eta_reference_seconds=600.0,
        initial_assignment_probability=0.5,
        tournament_size=3,
        elitism_count=2,
    )

    assert config.population_size == 50
    assert config.elitism_count == 2


def test_envelope_wraps_plan_response():
    envelope = ResponsePlanEnvelopeResponse(plan=make_plan_response())

    assert envelope.plan is not None
    assert envelope.plan.plan_id == 7


def test_envelope_allows_null_plan():
    envelope = ResponsePlanEnvelopeResponse(plan=None)

    assert envelope.model_dump(mode="json") == {"plan": None}


def test_path_coordinates_preserve_order():
    coordinates = [
        CoordinateResponse(latitude=1.0, longitude=1.0),
        CoordinateResponse(latitude=2.0, longitude=2.0),
    ]
    route = ResponsePlanRouteResponse(
        status=RouteStatus.REACHABLE,
        eta_seconds=1.0,
        distance_meters=1.0,
        node_path=[1, 2],
        path_coordinates=coordinates,
    )

    assert [(c.latitude, c.longitude) for c in route.path_coordinates] == [(1.0, 1.0), (2.0, 2.0)]


def test_uncovered_targets_preserve_order():
    response = make_plan_response(
        uncovered_targets=[
            ResponsePlanTargetResponse(
                response_target_id=5, target_type="active_fire", priority_score=0.4, latitude=1.0, longitude=1.0
            ),
            ResponsePlanTargetResponse(
                response_target_id=2, target_type="active_fire", priority_score=0.2, latitude=2.0, longitude=2.0
            ),
        ]
    )

    assert [t.response_target_id for t in response.uncovered_targets] == [5, 2]


def test_schema_module_does_not_import_calculators_or_repositories():
    module_path = Path(__file__).resolve().parents[3] / "src" / "api" / "schemas" / "response_plans.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_substrings = ("calculators", "repositories", "repository")
    offending = [
        module
        for module in imported_modules
        if any(substring in module for substring in forbidden_substrings)
    ]
    assert offending == []
