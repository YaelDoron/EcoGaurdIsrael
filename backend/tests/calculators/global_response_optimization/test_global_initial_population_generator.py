"""Tests for GlobalInitialPopulationGenerator (Stage 4, Task 12)."""
from __future__ import annotations

import random

import pytest

from src.calculators.global_response_optimization.global_initial_population_generator import (
    GlobalInitialPopulationGenerator,
)
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from tests.calculators.global_response_optimization.helpers import make_input, make_resource, make_route, make_target


def _problem(targets, resources, routes, active_fire_event_ids=(1, 2)):
    return build_global_optimization_problem(
        make_input(active_fire_event_ids=active_fire_event_ids, targets=targets, resources=resources, routes=routes)
    )


def test_population_size_matches_config():
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), (make_route("R1", 1, 10, eta_seconds=1.0),))
    config = GlobalResponseOptimizationConfig(population_size=6)

    population = GlobalInitialPopulationGenerator.generate(problem, config)

    assert len(population) == 6


def test_first_seed_is_all_idle():
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), (make_route("R1", 1, 10, eta_seconds=1.0),))
    config = GlobalResponseOptimizationConfig(population_size=4)

    population = GlobalInitialPopulationGenerator.generate(problem, config)

    assert population[0].genes == (None,)


def test_greedy_seed_prefers_the_higher_value_pair_under_contention():
    """Two resources, one target: R1 has the much better ETA - the greedy
    seed must claim R1 for the target, not R2."""
    problem = _problem(
        (make_target(1, 10, priority_score=100.0),),
        (make_resource("R1"), make_resource("R2")),
        (make_route("R1", 1, 10, eta_seconds=10.0), make_route("R2", 1, 10, eta_seconds=1000.0)),
    )
    config = GlobalResponseOptimizationConfig(population_size=3)

    population = GlobalInitialPopulationGenerator.generate(problem, config)
    greedy_seed = population[1]

    slot_id = problem.slots[0].slot_id
    r1_index = problem.resource_ids.index("R1")
    r2_index = problem.resource_ids.index("R2")
    assert greedy_seed.genes[r1_index] == slot_id
    assert greedy_seed.genes[r2_index] is None


def test_greedy_seed_never_assigns_the_same_slot_twice():
    problem = _problem(
        (make_target(1, 10),),
        (make_resource("R1"), make_resource("R2")),
        (make_route("R1", 1, 10, eta_seconds=10.0), make_route("R2", 1, 10, eta_seconds=20.0)),
    )
    config = GlobalResponseOptimizationConfig(population_size=3)

    population = GlobalInitialPopulationGenerator.generate(problem, config)
    greedy_seed = population[1]

    assigned = [gene for gene in greedy_seed.genes if gene is not None]
    assert len(assigned) == len(set(assigned))


def test_random_candidates_never_violate_feasibility_or_slot_uniqueness():
    problem = _problem(
        (make_target(1, 10), make_target(2, 20)),
        (make_resource("R1"), make_resource("R2")),
        (
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R1", 2, 20, eta_seconds=20.0),
            make_route("R2", 1, 10, eta_seconds=30.0),
        ),
    )
    config = GlobalResponseOptimizationConfig(population_size=20)

    population = GlobalInitialPopulationGenerator.generate(problem, config)

    for chromosome in population:
        assigned = [gene for gene in chromosome.genes if gene is not None]
        assert len(assigned) == len(set(assigned))
        for resource_id, slot_id in zip(chromosome.resource_ids, chromosome.genes):
            if slot_id is not None:
                assert slot_id in problem.feasible_slot_ids_by_resource[resource_id]


def test_generation_is_deterministic_for_the_same_seed():
    problem = _problem(
        (make_target(1, 10), make_target(2, 20)),
        (make_resource("R1"), make_resource("R2")),
        (make_route("R1", 1, 10, eta_seconds=10.0), make_route("R2", 2, 20, eta_seconds=20.0)),
    )
    config = GlobalResponseOptimizationConfig(population_size=10, random_seed=7)

    first = GlobalInitialPopulationGenerator.generate(problem, config, rng=random.Random(7))
    second = GlobalInitialPopulationGenerator.generate(problem, config, rng=random.Random(7))

    assert first == second


def test_rejects_invalid_arguments():
    problem = _problem((make_target(1, 10),), (make_resource("R1"),), ())
    with pytest.raises(ValueError):
        GlobalInitialPopulationGenerator.generate("not-a-problem", GlobalResponseOptimizationConfig())
    with pytest.raises(ValueError):
        GlobalInitialPopulationGenerator.generate(problem, "not-a-config")
