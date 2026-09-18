"""Task 32: single-incident compatibility - with exactly one active
FireEvent, the Global GA should reduce to a valid single-event allocation,
semantically comparable to the legacy per-event GA on the same scenario.
Chromosome representations differ deliberately (Task 32); only outcomes
(coverage, feasibility, resource uniqueness, score/ETA quality, and - where
the optimum is unique - the actual assignments) are compared.
"""
from __future__ import annotations

from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.calculators.response_optimization.genetic_optimizer import GeneticResponsePlanOptimizer
from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.models.optimization_resource import OptimizationResource
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.response_target_type import ResponseTargetType
from tests.calculators.global_response_optimization.helpers import make_input, make_resource, make_route, make_target


def test_single_incident_global_result_matches_legacy_optimizer_on_a_unique_optimum_scenario():
    # Scenario with a single clear optimum: NEAR should cover target 10, FAR should cover target 11.
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(
            make_target(1, 10, target_order=1, priority_score=100.0, target_type=ResponseTargetType.ACTIVE_FIRE),
            make_target(
                1, 11, target_order=2, priority_score=100.0,
                target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30,
            ),
        ),
        resources=(make_resource("NEAR"), make_resource("FAR")),
        routes=(
            make_route("NEAR", 1, 10, eta_seconds=20.0),
            make_route("NEAR", 1, 11, eta_seconds=400.0),
            make_route("FAR", 1, 10, eta_seconds=500.0),
            make_route("FAR", 1, 11, eta_seconds=60.0),
        ),
    )
    global_config = GlobalResponseOptimizationConfig(population_size=40, generation_count=60, random_seed=42)
    global_result = GlobalResponseOptimizationService().optimize(global_input, global_config)

    legacy_input = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=100,
        route_planning_run_id=200,
        targets=(
            OptimizationTarget(response_target_id=10, target_order=1, target_type=ResponseTargetType.ACTIVE_FIRE, priority_score=100.0),
            OptimizationTarget(response_target_id=11, target_order=2, target_type=ResponseTargetType.PREDICTED_RISK, priority_score=100.0),
        ),
        resources=(OptimizationResource("NEAR"), OptimizationResource("FAR")),
        route_options=(
            OptimizationRouteOption(route_result_id=1, resource_id="NEAR", response_target_id=10, is_reachable=True, travel_time_seconds=20.0, distance_meters=200.0),
            OptimizationRouteOption(route_result_id=2, resource_id="NEAR", response_target_id=11, is_reachable=True, travel_time_seconds=400.0, distance_meters=4000.0),
            OptimizationRouteOption(route_result_id=3, resource_id="FAR", response_target_id=10, is_reachable=True, travel_time_seconds=500.0, distance_meters=5000.0),
            OptimizationRouteOption(route_result_id=4, resource_id="FAR", response_target_id=11, is_reachable=True, travel_time_seconds=60.0, distance_meters=600.0),
        ),
    )
    legacy_config = ResponseOptimizationConfig(population_size=40, generation_count=60, random_seed=42)
    legacy_result = GeneticResponsePlanOptimizer().optimize(legacy_input, legacy_config)

    # Outcome comparison (Task 32): coverage, feasibility, uniqueness, and -
    # since this scenario has one unique optimum - the same assignments.
    assert global_result.coverage_score == legacy_result.score.coverage_score == 100.0
    global_assignment = {a.resource_id: a.response_target_id for a in global_result.actions}
    legacy_assignment = {a.resource_id: a.response_target_id for a in legacy_result.actions}
    assert global_assignment == legacy_assignment == {"NEAR": 10, "FAR": 11}

    resource_ids = [a.resource_id for a in global_result.actions]
    assert len(resource_ids) == len(set(resource_ids))


def test_single_incident_global_input_reduces_to_resources_times_that_events_targets():
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10), make_target(1, 11)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R1", 1, 11, eta_seconds=20.0),
            make_route("R2", 1, 10, eta_seconds=30.0),
            make_route("R2", 1, 11, eta_seconds=40.0),
        ),
    )

    result = GlobalResponseOptimizationService().optimize(
        global_input, GlobalResponseOptimizationConfig(population_size=20, generation_count=20, random_seed=1)
    )

    assert len(result.event_results) == 1
    (event_result,) = result.event_results
    assert event_result.fire_event_id == 1
    assert set(a.fire_event_id for a in result.actions) <= {1}
