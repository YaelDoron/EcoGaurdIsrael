"""Acceptance tests for GlobalResponseOptimizationService (Stage 4, Tasks
13-15, 23-32, 35-36).
"""
from __future__ import annotations

import itertools
import time

import pytest

from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.calculators.global_response_optimization.global_response_plan_scorer import GlobalResponsePlanScorer
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.resource_status import ResourceStatus
from tests.calculators.global_response_optimization.helpers import make_demand, make_input, make_resource, make_route, make_target

RELIABLE_CONFIG = GlobalResponseOptimizationConfig(population_size=40, generation_count=60, random_seed=42)


def _service() -> GlobalResponseOptimizationService:
    return GlobalResponseOptimizationService()


# ---------------------------------------------------------------------------
# Task 13 - nearest/lowest-ETA resource, uncontested
# ---------------------------------------------------------------------------


def test_uncontested_target_receives_the_lowest_eta_feasible_resource():
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=100.0),),
        resources=(make_resource("NEAR"), make_resource("FAR")),
        routes=(
            make_route("NEAR", 1, 10, eta_seconds=30.0),
            make_route("FAR", 1, 10, eta_seconds=3000.0),
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    (action,) = result.actions
    assert action.resource_id == "NEAR"


# ---------------------------------------------------------------------------
# Task 14 - global competition across two events
# ---------------------------------------------------------------------------


def test_global_competition_produces_the_globally_better_allocation():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=2.0),
            make_route("R1", 2, 20, eta_seconds=3.0),
            make_route("R2", 1, 10, eta_seconds=30.0),
            make_route("R2", 2, 20, eta_seconds=8.0),
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.response_target_id for action in result.actions}
    assert assignment == {"R1": 10, "R2": 20}


# ---------------------------------------------------------------------------
# Task 15 - farther-station resource genuinely usable
# ---------------------------------------------------------------------------


def test_farther_resource_covers_a_second_incident_rather_than_leaving_it_uncovered():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(make_resource("LOCAL"), make_resource("FARTHER")),
        routes=(
            make_route("LOCAL", 1, 10, eta_seconds=30.0),
            make_route("FARTHER", 2, 20, eta_seconds=1200.0),  # far, but the only option for event 2
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.response_target_id for action in result.actions}
    assert assignment == {"LOCAL": 10, "FARTHER": 20}


# ---------------------------------------------------------------------------
# Task 23 - resource global uniqueness
# ---------------------------------------------------------------------------


def test_shared_resource_never_appears_in_two_fire_events_result():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0), make_route("R1", 2, 20, eta_seconds=15.0)),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = [action.resource_id for action in result.actions]
    assert len(resource_ids) == len(set(resource_ids))
    assert len(result.actions) <= 1


# ---------------------------------------------------------------------------
# Task 24 - cross-event reassignment when globally better
# ---------------------------------------------------------------------------


def test_resource_near_event_a_is_assigned_to_event_b_when_globally_optimal():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            # R1 is geographically "near" A (feasible with a mediocre ETA there)
            # but is EXCELLENT for B; R2 is excellent for A but infeasible for B.
            make_route("R1", 1, 10, eta_seconds=500.0),
            make_route("R1", 2, 20, eta_seconds=5.0),
            make_route("R2", 1, 10, eta_seconds=5.0),
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assignment = {action.resource_id: action.response_target_id for action in result.actions}
    assert assignment == {"R1": 20, "R2": 10}


# ---------------------------------------------------------------------------
# Task 25 - unreachable pairs never selected
# ---------------------------------------------------------------------------


def test_unreachable_pair_is_never_selected_in_the_final_result():
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=100.0), make_target(1, 11, priority_score=100.0)),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0),),  # R1 -> target 11 is unreachable (no route)
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    for action in result.actions:
        assert not (action.resource_id == "R1" and action.response_target_id == 11)


# ---------------------------------------------------------------------------
# Task 26 - insufficient resources
# ---------------------------------------------------------------------------


def test_insufficient_resources_leaves_at_least_one_slot_uncovered_without_duplication():
    """Task 26: 3 desired suppression slots on one incident, only 2 feasible
    resources -> at most 2 covered, no duplication, no fabricated resource."""
    demand = make_demand(1, minimum_resources=0, desired_resources=3)
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10, priority_score=100.0),),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R2", 1, 10, eta_seconds=15.0),
        ),
        incident_demands=(demand,),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    resource_ids = [action.resource_id for action in result.actions]
    assert len(resource_ids) == len(set(resource_ids)) <= 2
    assert len(result.uncovered_slot_ids) >= 1


# ---------------------------------------------------------------------------
# Task 27 - more resources than slots
# ---------------------------------------------------------------------------


def test_more_resources_than_slots_leaves_the_rest_idle():
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10), make_target(1, 11)),
        resources=tuple(make_resource(f"R{i}") for i in range(5)),
        routes=tuple(
            make_route(f"R{i}", 1, target_id, eta_seconds=10.0 * (i + 1))
            for i in range(5)
            for target_id in (10, 11)
        ),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assert len(result.actions) <= 2
    assigned = {action.resource_id for action in result.actions}
    assert len(assigned) <= 2


# ---------------------------------------------------------------------------
# Task 28 - committed resource remains evaluable, never mutated
# ---------------------------------------------------------------------------


def test_committed_resource_is_still_evaluated_and_metadata_survives_untouched():
    committed_resource = make_resource(
        "R1", current_commitment_fire_event_id=1, current_commitment_response_plan_id=99
    )
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(committed_resource,),
        routes=(make_route("R1", 1, 10, eta_seconds=5.0), make_route("R1", 2, 20, eta_seconds=5.0)),
    )

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    # The original input object's commitment metadata is untouched (Stage 4 never writes anywhere).
    original_resource = next(r for r in global_input.resources if r.resource_id == "R1")
    assert original_resource.current_commitment_fire_event_id == 1
    assert original_resource.current_commitment_response_plan_id == 99
    # R1 was still genuinely evaluable for BOTH events despite the commitment.
    assert len(result.actions) == 1


# ---------------------------------------------------------------------------
# Task 29 - small exact-oracle test
# ---------------------------------------------------------------------------


def _exhaustive_best_fitness(problem, config) -> float:
    """Test-only exhaustive enumeration of every feasible chromosome - never used in production."""
    scorer = GlobalResponsePlanScorer(config)
    choices_per_resource = [
        [None, *problem.feasible_slot_ids_by_resource[resource_id]] for resource_id in problem.resource_ids
    ]
    best = 0.0
    for combo in itertools.product(*choices_per_resource):
        assigned = [slot_id for slot_id in combo if slot_id is not None]
        if len(assigned) != len(set(assigned)):
            continue
        chromosome = GlobalResponsePlanChromosome(problem.resource_ids, combo)
        score = scorer.evaluate(problem, chromosome)
        best = max(best, score.fitness_score)
    return best


def test_ga_reaches_the_exhaustively_computed_optimum_on_a_small_fixture():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=80.0), make_target(2, 20, priority_score=120.0)),
        resources=(make_resource("R1"), make_resource("R2"), make_resource("R3")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=40.0),
            make_route("R1", 2, 20, eta_seconds=600.0),
            make_route("R2", 1, 10, eta_seconds=500.0),
            make_route("R2", 2, 20, eta_seconds=25.0),
            make_route("R3", 1, 10, eta_seconds=90.0),
            make_route("R3", 2, 20, eta_seconds=90.0),
        ),
    )
    problem = build_global_optimization_problem(global_input)
    exhaustive_best = _exhaustive_best_fitness(problem, RELIABLE_CONFIG)

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assert result.fitness_score == pytest.approx(exhaustive_best, abs=1e-9)


# ---------------------------------------------------------------------------
# Task 30/31 - reproducibility and different seeds
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_results():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R1", 2, 20, eta_seconds=20.0),
            make_route("R2", 1, 10, eta_seconds=30.0),
            make_route("R2", 2, 20, eta_seconds=15.0),
        ),
    )

    first = _service().optimize(global_input, RELIABLE_CONFIG)
    second = _service().optimize(global_input, RELIABLE_CONFIG)

    assert first.actions == second.actions
    assert first.fitness_score == second.fitness_score
    assert first.coverage_score == second.coverage_score


def test_different_seeds_each_remain_valid_and_individually_reproducible():
    global_input = make_input(
        active_fire_event_ids=(1, 2),
        targets=(make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R1", 2, 20, eta_seconds=20.0),
            make_route("R2", 1, 10, eta_seconds=30.0),
            make_route("R2", 2, 20, eta_seconds=15.0),
        ),
    )
    config_a = GlobalResponseOptimizationConfig(population_size=10, generation_count=10, random_seed=1)
    config_b = GlobalResponseOptimizationConfig(population_size=10, generation_count=10, random_seed=2)

    result_a1 = _service().optimize(global_input, config_a)
    result_a2 = _service().optimize(global_input, config_a)
    result_b1 = _service().optimize(global_input, config_b)
    result_b2 = _service().optimize(global_input, config_b)

    assert result_a1.actions == result_a2.actions
    assert result_b1.actions == result_b2.actions
    resource_ids_a = [a.resource_id for a in result_a1.actions]
    resource_ids_b = [a.resource_id for a in result_b1.actions]
    assert len(resource_ids_a) == len(set(resource_ids_a))
    assert len(resource_ids_b) == len(set(resource_ids_b))


# ---------------------------------------------------------------------------
# Task 35/36 - search-space reporting and performance guard
# ---------------------------------------------------------------------------


def test_search_space_metadata_is_reportable():
    global_input = make_input(
        active_fire_event_ids=(1,),
        targets=(make_target(1, 10),),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(make_route("R1", 1, 10, eta_seconds=10.0),),
    )
    problem = build_global_optimization_problem(global_input)

    resource_count = len(problem.resource_ids)
    slot_count = len(problem.slots)
    potential_pairs = resource_count * slot_count
    feasible_pairs = sum(len(slots) for slots in problem.feasible_slot_ids_by_resource.values())

    assert resource_count == 2
    assert slot_count == 1
    assert potential_pairs == 2
    assert feasible_pairs == 1


def test_moderately_larger_synthetic_scenario_completes_and_preserves_invariants():
    resource_count = 16
    slot_count = 14
    fire_event_ids = (1, 2, 3)
    targets = tuple(
        make_target(fire_event_ids[i % 3], 100 + i, priority_score=50.0 + i) for i in range(slot_count)
    )
    resources = tuple(make_resource(f"R{i}") for i in range(resource_count))
    routes = tuple(
        make_route(f"R{i}", fire_event_ids[j % 3], 100 + j, eta_seconds=float(10 * (i + 1) + j))
        for i in range(resource_count)
        for j in range(slot_count)
        if (i + j) % 3 != 0  # leave some pairs infeasible, still plenty of coverage
    )
    global_input = make_input(
        active_fire_event_ids=fire_event_ids, targets=targets, resources=resources, routes=routes
    )
    config = GlobalResponseOptimizationConfig(population_size=20, generation_count=15, random_seed=99)

    started = time.monotonic()
    result = _service().optimize(global_input, config)
    elapsed = time.monotonic() - started

    resource_ids = [action.resource_id for action in result.actions]
    assert len(resource_ids) == len(set(resource_ids))
    assert len(result.actions) <= min(resource_count, slot_count)
    assert elapsed < 30.0  # generous, non-brittle guard - just proves it terminates promptly


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_no_active_events_or_targets_produces_an_empty_valid_result():
    global_input = make_input(active_fire_event_ids=(), targets=(), resources=(), routes=())

    result = _service().optimize(global_input, RELIABLE_CONFIG)

    assert result.actions == ()
    assert result.event_results == ()


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        _service().optimize("not-an-input", RELIABLE_CONFIG)
    global_input = make_input(active_fire_event_ids=(1,), targets=(make_target(1, 10),), resources=(), routes=())
    with pytest.raises(ValueError):
        _service().optimize(global_input, "not-a-config")
