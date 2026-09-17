"""Tests for US 5.2 response-optimization pure domain models."""
from __future__ import annotations

from datetime import datetime, timezone
import ast
from pathlib import Path

import pytest

from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.models import (
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    ResponseAction,
    ResponseOptimizationInput,
    ResponsePlan,
    ResponsePlanChromosome,
    ResponsePlanStatus,
    ResponseTargetType,
)

AS_OF = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def target(target_id: int = 10, order: int = 0, priority: float = 150.0) -> OptimizationTarget:
    return OptimizationTarget(
        response_target_id=target_id,
        target_order=order,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        priority_score=priority,
    )


def resource(resource_id: int | str = "TRUCK-A") -> OptimizationResource:
    return OptimizationResource(resource_id=resource_id)


def route(
    route_id: int = 100,
    resource_id: int | str = "TRUCK-A",
    target_id: int = 10,
    *,
    reachable: bool = True,
) -> OptimizationRouteOption:
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=300.0 if reachable else None,
        distance_meters=2500.0 if reachable else None,
    )


def action(resource_id: int | str = "TRUCK-A", target_id: int = 10, route_id: int = 100) -> ResponseAction:
    return ResponseAction(resource_id=resource_id, response_target_id=target_id, route_result_id=route_id)


def test_optimization_target_valid_construction_and_deterministic_ordering():
    first = target(20, order=1, priority=90.0)
    second = target(10, order=0, priority=150.0)

    assert sorted((first, second), key=lambda item: item.ordering_key) == [second, first]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_target_id": 0},
        {"target_order": -1},
        {"priority_score": -0.1},
        {"priority_score": float("nan")},
        {"target_type": "active_fire"},
    ],
)
def test_optimization_target_rejects_invalid_values(kwargs):
    values = {
        "response_target_id": 10,
        "target_order": 0,
        "target_type": ResponseTargetType.ACTIVE_FIRE,
        "priority_score": 1.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        OptimizationTarget(**values)


def test_optimization_resource_valid_construction_normalizes_string_id():
    assert OptimizationResource(" TRUCK-A ").resource_id == "TRUCK-A"


@pytest.mark.parametrize("resource_id", [0, "", "   ", True])
def test_optimization_resource_rejects_invalid_id(resource_id):
    with pytest.raises(ValueError):
        OptimizationResource(resource_id)


def test_optimization_route_option_reachable_valid_option():
    option = route()

    assert option.is_reachable is True
    assert option.travel_time_seconds == 300.0
    assert option.distance_meters == 2500.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"travel_time_seconds": -1.0},
        {"distance_meters": -1.0},
        {"travel_time_seconds": None},
        {"distance_meters": None},
    ],
)
def test_optimization_route_option_rejects_invalid_reachable_metrics(kwargs):
    values = {
        "route_result_id": 100,
        "resource_id": "TRUCK-A",
        "response_target_id": 10,
        "is_reachable": True,
        "travel_time_seconds": 300.0,
        "distance_meters": 2500.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        OptimizationRouteOption(**values)


def test_optimization_route_option_non_reachable_allows_none_metrics():
    option = route(reachable=False)

    assert option.travel_time_seconds is None
    assert option.distance_meters is None


@pytest.mark.parametrize("field_name", ["travel_time_seconds", "distance_meters"])
def test_optimization_route_option_non_reachable_rejects_metrics(field_name):
    values = {
        "route_result_id": 100,
        "resource_id": "TRUCK-A",
        "response_target_id": 10,
        "is_reachable": False,
        "travel_time_seconds": None,
        "distance_meters": None,
    }
    values[field_name] = 10.0

    with pytest.raises(ValueError):
        OptimizationRouteOption(**values)


def test_response_optimization_input_valid_construction_and_deterministic_normalization():
    input_data = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(20, 1), target(10, 0)),
        resources=(resource("TRUCK-B"), resource("TRUCK-A")),
        route_options=(
            route(300, "TRUCK-B", 20),
            route(100, "TRUCK-A", 10),
            route(200, "TRUCK-A", 20),
        ),
    )

    assert [item.response_target_id for item in input_data.targets] == [10, 20]
    assert [item.resource_id for item in input_data.resources] == ["TRUCK-A", "TRUCK-B"]
    assert [item.route_result_id for item in input_data.route_options] == [100, 200, 300]


@pytest.mark.parametrize(
    "overrides",
    [
        {"targets": (target(10, 0), target(10, 1))},
        {"targets": (target(10, 0), target(20, 0))},
        {"resources": (resource("TRUCK-A"), resource("TRUCK-A"))},
        {"route_options": (route(100, "TRUCK-A", 10), route(100, "TRUCK-B", 20))},
        {"route_options": (route(100, "TRUCK-A", 10), route(101, "TRUCK-A", 10))},
        {"route_options": (route(100, "MISSING", 10),)},
        {"route_options": (route(100, "TRUCK-A", 999),)},
    ],
)
def test_response_optimization_input_rejects_invalid_cross_references_and_duplicates(overrides):
    values = {
        "fire_event_id": 1,
        "response_target_set_id": 2,
        "route_planning_run_id": 3,
        "targets": (target(10, 0), target(20, 1)),
        "resources": (resource("TRUCK-A"), resource("TRUCK-B")),
        "route_options": (route(100, "TRUCK-A", 10), route(200, "TRUCK-B", 20)),
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        ResponseOptimizationInput(**values)


def test_response_optimization_input_allows_empty_resources_and_routes():
    input_data = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(10, 0),),
        resources=(),
        route_options=(),
    )

    assert input_data.resources == ()
    assert input_data.route_options == ()


def test_response_optimization_input_allows_zero_reachable_assignments():
    input_data = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(10, 0),),
        resources=(resource("TRUCK-A"),),
        route_options=(route(100, "TRUCK-A", 10, reachable=False),),
    )

    assert all(not option.is_reachable for option in input_data.route_options)


def test_response_optimization_input_requires_only_normalized_planning_facts():
    input_data = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(10, 0),),
        resources=(resource("TRUCK-A"),),
        route_options=(route(100, "TRUCK-A", 10),),
    )

    assert not hasattr(input_data, "severity_assessment")
    assert not hasattr(input_data, "spread_prediction")
    assert not hasattr(input_data, "weather_observation")
    assert not hasattr(input_data, "road_graph")
    assert not hasattr(input_data, "dijkstra")


def test_response_action_valid_construction_and_invalid_ids():
    assert action().resource_id == "TRUCK-A"

    with pytest.raises(ValueError):
        ResponseAction(resource_id="", response_target_id=10, route_result_id=100)
    with pytest.raises(ValueError):
        ResponseAction(resource_id="TRUCK-A", response_target_id=0, route_result_id=100)
    with pytest.raises(ValueError):
        ResponseAction(resource_id="TRUCK-A", response_target_id=10, route_result_id=0)


def test_response_plan_chromosome_exported_and_target_indexed():
    chromosome = ResponsePlanChromosome(("TRUCK-A", None, "TRUCK-B"))

    assert chromosome.genes == ("TRUCK-A", None, "TRUCK-B")


def test_response_plan_status_exact_expected_values():
    assert ResponsePlanStatus.COMPLETE.value == "complete"
    assert ResponsePlanStatus.PARTIAL.value == "partial"
    assert ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS.value == "no_feasible_assignments"


def make_plan(**overrides) -> ResponsePlan:
    values = {
        "fire_event_id": 1,
        "response_target_set_id": 2,
        "route_planning_run_id": 3,
        "generated_at": AS_OF,
        "status": ResponsePlanStatus.COMPLETE,
        "methodology": "GENETIC_RESOURCE_ALLOCATION",
        "methodology_version": "1.0",
        "random_seed": 42,
        "actions": (action("TRUCK-A", 10, 100),),
        "uncovered_target_ids": (),
        "plan_score": 10.0,
        "coverage_score": 1.0,
        "average_eta_seconds": 300.0,
    }
    values.update(overrides)
    return ResponsePlan(**values)


def test_response_plan_valid_construction_and_deterministic_action_ordering():
    plan = make_plan(actions=(action("TRUCK-B", 20, 200), action("TRUCK-A", 10, 100)))

    assert [item.resource_id for item in plan.actions] == ["TRUCK-A", "TRUCK-B"]


# ---------------------------------------------------------------------------
# FND-04: optimization_config (exact GA configuration snapshot)
# ---------------------------------------------------------------------------


def test_response_plan_without_optimization_config_defaults_to_none():
    plan = make_plan()

    assert plan.optimization_config is None


def test_response_plan_accepts_a_full_optimization_config():
    config = ResponseOptimizationConfig(
        population_size=30,
        generation_count=55,
        mutation_rate=0.12,
        crossover_rate=0.65,
        random_seed=42,
        eta_reference_seconds=750.0,
        initial_assignment_probability=0.6,
        tournament_size=3,
        elitism_count=2,
    )

    plan = make_plan(random_seed=42, optimization_config=config)

    assert plan.optimization_config == config
    assert plan.optimization_config is config


def test_response_plan_rejects_non_config_optimization_config():
    with pytest.raises(ValueError):
        make_plan(optimization_config="not-a-config")


def test_response_plan_rejects_optimization_config_seed_mismatch():
    config = ResponseOptimizationConfig(random_seed=1)

    with pytest.raises(ValueError):
        make_plan(random_seed=2, optimization_config=config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"generated_at": datetime(2026, 9, 16, 10, 0)},
        {"actions": (action("TRUCK-A", 10, 100), action("TRUCK-A", 20, 200))},
        {"actions": (action("TRUCK-A", 10, 100), action("TRUCK-B", 10, 200)), "uncovered_target_ids": (10,)},
        {"status": ResponsePlanStatus.COMPLETE, "uncovered_target_ids": (20,)},
        {"status": ResponsePlanStatus.PARTIAL, "actions": (), "uncovered_target_ids": (20,)},
        {"status": ResponsePlanStatus.PARTIAL, "actions": (action("TRUCK-A", 10, 100),), "uncovered_target_ids": ()},
        {"status": ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS, "actions": (action("TRUCK-A", 10, 100),)},
    ],
)
def test_response_plan_rejects_invalid_invariants(overrides):
    with pytest.raises(ValueError):
        make_plan(**overrides)


def test_response_plan_allows_no_feasible_assignments_with_uncovered_targets():
    plan = make_plan(
        status=ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
        actions=(),
        uncovered_target_ids=(10, 20),
        plan_score=0.0,
        coverage_score=0.0,
        average_eta_seconds=None,
    )

    assert plan.actions == ()
    assert plan.uncovered_target_ids == (10, 20)


def test_response_optimization_pure_modules_do_not_import_out_of_scope_systems():
    forbidden_fragments = (
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "ResponseTargetGenerationAgent",
        "OperationalRefreshOrchestrator",
        "FireSpreadCalculator",
        "ResponseTargetCalculator",
        "RoadNetworkFetcher",
        "Dijkstra",
        "simulation",
        "external",
        "sqlalchemy",
        "repositories",
        "database",
    )
    paths = [
        Path("backend/src/models/optimization_target.py"),
        Path("backend/src/models/optimization_resource.py"),
        Path("backend/src/models/optimization_route_option.py"),
        Path("backend/src/models/response_optimization_input.py"),
        Path("backend/src/models/response_action.py"),
        Path("backend/src/models/response_plan_chromosome.py"),
        Path("backend/src/models/plan_score_breakdown.py"),
        Path("backend/src/models/response_plan.py"),
        Path("backend/src/calculators/response_optimization/response_optimization_config.py"),
        Path("backend/src/calculators/response_optimization/response_plan_scorer.py"),
        Path("backend/src/calculators/response_optimization/feasible_route_lookup.py"),
        Path("backend/src/calculators/response_optimization/chromosome_decoder.py"),
        Path("backend/src/calculators/response_optimization/initial_population_generator.py"),
        Path("backend/src/calculators/response_optimization/genetic_optimizer.py"),
    ]
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if any(fragment in module for fragment in forbidden_fragments):
                violations.append((str(path), module))

    assert violations == []
