"""Tests for US 5.5 Task 1 response-plan-details read models (DTOs).

These models are a pure, read-only presentation contract: they must not
recompute routing, optimization, or target-priority logic, so this suite
also asserts the module stays free of imports into those subsystems.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models import (
    BaselineComparisonDetails,
    OptimizationConfigDetails,
    ResponseActionDetails,
    ResponsePlanDetails,
    ResponseTargetType,
)

AS_OF = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)


def action(**overrides) -> ResponseActionDetails:
    values = {
        "resource_id": "TRUCK-A",
        "station_id": "STATION-1",
        "response_target_id": 10,
        "target_type": ResponseTargetType.ACTIVE_FIRE.value,
        "target_priority": 150.0,
        "eta_seconds": 300.0,
        "route_distance_meters": 2500.0,
        "node_path": (1, 2, 3),
    }
    values.update(overrides)
    return ResponseActionDetails(**values)


def baseline_comparison(**overrides) -> BaselineComparisonDetails:
    values = {
        "baseline_score": 8.0,
        "baseline_coverage_score": 0.8,
        "baseline_average_eta_seconds": 400.0,
        "score_difference": 2.0,
        "improvement_percentage": 25.0,
    }
    values.update(overrides)
    return BaselineComparisonDetails(**values)


def optimization_config(**overrides) -> OptimizationConfigDetails:
    values = {
        "population_size": 24,
        "generation_count": 40,
        "mutation_rate": 0.08,
        "crossover_rate": 0.75,
        "eta_reference_seconds": 900.0,
        "initial_assignment_probability": 0.75,
        "tournament_size": 2,
        "elitism_count": 1,
    }
    values.update(overrides)
    return OptimizationConfigDetails(**values)


def plan(**overrides) -> ResponsePlanDetails:
    values = {
        "plan_id": 1,
        "fire_event_id": 2,
        "response_target_set_id": 3,
        "route_planning_run_id": 4,
        "generated_at": AS_OF,
        "methodology": "GENETIC_RESOURCE_ALLOCATION",
        "methodology_version": "1.0",
        "random_seed": 42,
        "is_current": True,
        "plan_score": 10.0,
        "coverage_score": 1.0,
        "average_eta_seconds": 300.0,
        "actions": (action(),),
        "uncovered_target_ids": (),
        "baseline_comparison": baseline_comparison(),
        "optimization_config": optimization_config(),
    }
    values.update(overrides)
    return ResponsePlanDetails(**values)


# --- ResponseActionDetails -------------------------------------------------


def test_response_action_details_valid_construction():
    details = action()

    assert details.resource_id == "TRUCK-A"
    assert details.station_id == "STATION-1"
    assert details.node_path == (1, 2, 3)


def test_response_action_details_allows_missing_route_data():
    details = action(eta_seconds=None, route_distance_meters=None, node_path=None)

    assert details.eta_seconds is None
    assert details.route_distance_meters is None
    assert details.node_path is None


def test_response_action_details_normalizes_node_path_to_tuple():
    details = action(node_path=[1, 2, 3])

    assert details.node_path == (1, 2, 3)
    assert isinstance(details.node_path, tuple)


def test_response_action_details_is_immutable():
    details = action()

    with pytest.raises(FrozenInstanceError):
        details.resource_id = "TRUCK-B"


@pytest.mark.parametrize(
    "overrides",
    [
        {"resource_id": ""},
        {"resource_id": "   "},
        {"station_id": ""},
        {"response_target_id": 0},
        {"response_target_id": -1},
        {"target_type": "not_a_real_type"},
        {"target_priority": float("nan")},
        {"eta_seconds": -1.0},
        {"route_distance_meters": -1.0},
        {"node_path": (0, 1)},
        {"node_path": (1, -2)},
    ],
)
def test_response_action_details_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        action(**overrides)


# --- OptimizationConfigDetails ----------------------------------------------


def test_optimization_config_details_valid_construction():
    config = optimization_config()

    assert config.population_size == 24
    assert config.elitism_count == 1


def test_optimization_config_details_is_immutable():
    config = optimization_config()

    with pytest.raises(FrozenInstanceError):
        config.population_size = 1


def test_optimization_config_details_allows_zero_elitism_count():
    config = optimization_config(elitism_count=0)

    assert config.elitism_count == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"population_size": 0},
        {"population_size": -1},
        {"generation_count": 0},
        {"mutation_rate": float("nan")},
        {"crossover_rate": float("inf")},
        {"eta_reference_seconds": float("nan")},
        {"initial_assignment_probability": float("nan")},
        {"tournament_size": 0},
        {"elitism_count": -1},
        {"elitism_count": True},
    ],
)
def test_optimization_config_details_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        optimization_config(**overrides)


# --- BaselineComparisonDetails ----------------------------------------------


def test_baseline_comparison_details_valid_construction():
    comparison = baseline_comparison()

    assert comparison.baseline_score == 8.0
    assert comparison.improvement_percentage == 25.0


def test_baseline_comparison_details_allows_missing_optional_fields():
    comparison = baseline_comparison(baseline_average_eta_seconds=None, improvement_percentage=None)

    assert comparison.baseline_average_eta_seconds is None
    assert comparison.improvement_percentage is None


def test_baseline_comparison_details_is_immutable():
    comparison = baseline_comparison()

    with pytest.raises(FrozenInstanceError):
        comparison.baseline_score = 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"baseline_score": float("nan")},
        {"baseline_coverage_score": float("inf")},
        {"baseline_average_eta_seconds": -1.0},
        {"score_difference": float("nan")},
        {"improvement_percentage": float("nan")},
    ],
)
def test_baseline_comparison_details_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        baseline_comparison(**overrides)


# --- ResponsePlanDetails -----------------------------------------------------


def test_response_plan_details_valid_construction():
    result = plan()

    assert result.plan_id == 1
    assert result.is_current is True
    assert result.actions == (action(),)
    assert result.baseline_comparison is not None
    assert result.random_seed == 42
    assert result.optimization_config is not None
    assert result.optimization_config.population_size == 24


def test_response_plan_details_allows_missing_baseline_comparison():
    result = plan(baseline_comparison=None)

    assert result.baseline_comparison is None


def test_response_plan_details_allows_none_optimization_config_for_legacy_rows():
    result = plan(optimization_config=None)

    assert result.optimization_config is None
    assert result.random_seed == 42


def test_response_plan_details_allows_empty_actions_and_uncovered_targets():
    result = plan(actions=(), uncovered_target_ids=(20, 30))

    assert result.actions == ()
    assert result.uncovered_target_ids == (20, 30)


def test_response_plan_details_allows_none_average_eta_seconds():
    result = plan(average_eta_seconds=None)

    assert result.average_eta_seconds is None


def test_response_plan_details_normalizes_lists_to_tuples():
    result = plan(actions=[action()], uncovered_target_ids=[20, 30])

    assert isinstance(result.actions, tuple)
    assert isinstance(result.uncovered_target_ids, tuple)


def test_response_plan_details_is_immutable():
    result = plan()

    with pytest.raises(FrozenInstanceError):
        result.plan_score = 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"plan_id": 0},
        {"fire_event_id": -1},
        {"response_target_set_id": 0},
        {"route_planning_run_id": 0},
        {"generated_at": datetime(2026, 9, 16, 10, 0)},
        {"methodology": ""},
        {"methodology_version": ""},
        {"random_seed": "42"},
        {"random_seed": True},
        {"is_current": "true"},
        {"plan_score": float("nan")},
        {"coverage_score": float("inf")},
        {"average_eta_seconds": -1.0},
        {"actions": (object(),)},
        {"uncovered_target_ids": (0,)},
        {"baseline_comparison": object()},
        {"optimization_config": object()},
    ],
)
def test_response_plan_details_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


def test_response_plan_details_does_not_import_calculation_or_persistence_modules():
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
        "genetic_optimizer",
        "NodeMapping",
        "simulation",
        "external",
        "sqlalchemy",
        "repositories",
        "database",
        "calculators",
        "services",
    )
    path = (Path(__file__).resolve().parents[3] / "backend/src/models/response_plan_details.py")
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
