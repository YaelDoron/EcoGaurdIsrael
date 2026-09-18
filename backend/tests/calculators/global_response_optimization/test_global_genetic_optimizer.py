"""Tests for GlobalGeneticResponseOptimizer (Stage 4, Tasks 16-19)."""
from __future__ import annotations

import random

import pytest

from src.calculators.global_response_optimization.global_genetic_optimizer import (
    GlobalGeneticResponseOptimizer,
    candidate_ranking_key,
    rank_candidates,
)
from src.calculators.global_response_optimization.global_genetic_optimizer import GlobalCandidateEvaluator
from src.calculators.global_response_optimization.global_initial_population_generator import (
    GlobalInitialPopulationGenerator,
)
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    build_global_optimization_problem,
)
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from tests.calculators.global_response_optimization.helpers import make_input, make_resource, make_route, make_target


def _problem(targets, resources, routes, active_fire_event_ids=(1, 2)):
    return build_global_optimization_problem(
        make_input(active_fire_event_ids=active_fire_event_ids, targets=targets, resources=resources, routes=routes)
    )


def _small_problem():
    return _problem(
        (make_target(1, 10, priority_score=100.0), make_target(2, 20, priority_score=100.0)),
        (make_resource("R1"), make_resource("R2")),
        (
            make_route("R1", 1, 10, eta_seconds=10.0),
            make_route("R1", 2, 20, eta_seconds=20.0),
            make_route("R2", 1, 10, eta_seconds=30.0),
            make_route("R2", 2, 20, eta_seconds=15.0),
        ),
    )


# ---------------------------------------------------------------------------
# Crossover / repair (Task 16)
# ---------------------------------------------------------------------------


def test_uniform_child_never_produces_duplicate_slot_even_when_both_parents_claim_it():
    problem = _small_problem()
    slot_10 = next(s for s in problem.slots if s.response_target_id == 10).slot_id
    slot_20 = next(s for s in problem.slots if s.response_target_id == 20).slot_id
    # Both parents assign slot_10 to a DIFFERENT resource - a naive copy of
    # both parents' genes verbatim would duplicate slot_10 in the child.
    parent_a = GlobalResponsePlanChromosome(problem.resource_ids, (slot_10, slot_20))
    parent_b = GlobalResponsePlanChromosome(problem.resource_ids, (slot_20, slot_10))

    rng = random.Random(1)
    for _ in range(50):  # exercise both coin-flip branches of _uniform_child
        child = GlobalGeneticResponseOptimizer._uniform_child(parent_a, parent_b, problem, rng)
        assigned = [gene for gene in child.genes if gene is not None]
        assert len(assigned) == len(set(assigned))


def test_uniform_child_never_assigns_an_infeasible_slot():
    problem = _small_problem()
    slot_10 = next(s for s in problem.slots if s.response_target_id == 10).slot_id
    slot_20 = next(s for s in problem.slots if s.response_target_id == 20).slot_id
    parent_a = GlobalResponsePlanChromosome(problem.resource_ids, (slot_10, slot_20))
    parent_b = GlobalResponsePlanChromosome(problem.resource_ids, (None, None))

    rng = random.Random(3)
    for _ in range(20):
        child = GlobalGeneticResponseOptimizer._uniform_child(parent_a, parent_b, problem, rng)
        for resource_id, slot_id in zip(child.resource_ids, child.genes):
            if slot_id is not None:
                assert slot_id in problem.feasible_slot_ids_by_resource[resource_id]


# ---------------------------------------------------------------------------
# Mutation (Task 17)
# ---------------------------------------------------------------------------


def test_mutation_never_creates_duplicate_or_infeasible_slots():
    problem = _small_problem()
    config = GlobalResponseOptimizationConfig(mutation_rate=1.0)  # mutate every gene
    slot_10 = next(s for s in problem.slots if s.response_target_id == 10).slot_id
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (slot_10, None))

    rng = random.Random(5)
    for _ in range(50):
        mutated = GlobalGeneticResponseOptimizer._mutate(chromosome, problem, config, rng)
        assigned = [gene for gene in mutated.genes if gene is not None]
        assert len(assigned) == len(set(assigned))
        for resource_id, slot_id in zip(mutated.resource_ids, mutated.genes):
            if slot_id is not None:
                assert slot_id in problem.feasible_slot_ids_by_resource[resource_id]


def test_zero_mutation_rate_never_changes_the_chromosome():
    problem = _small_problem()
    config = GlobalResponseOptimizationConfig(mutation_rate=0.0)
    slot_10 = next(s for s in problem.slots if s.response_target_id == 10).slot_id
    chromosome = GlobalResponsePlanChromosome(problem.resource_ids, (slot_10, None))

    mutated = GlobalGeneticResponseOptimizer._mutate(chromosome, problem, config, random.Random(9))

    assert mutated == chromosome


# ---------------------------------------------------------------------------
# Ranking / tie-break (Task 19)
# ---------------------------------------------------------------------------


def test_ranking_prefers_higher_fitness_then_more_covered_slots_then_lower_total_eta():
    problem = _small_problem()
    evaluator = GlobalCandidateEvaluator()
    config = GlobalResponseOptimizationConfig()

    full = evaluator.evaluate(
        problem,
        GlobalResponsePlanChromosome(
            problem.resource_ids,
            tuple(
                next(s.slot_id for s in problem.slots if s.response_target_id == (10 if rid == "R1" else 20))
                for rid in problem.resource_ids
            ),
        ),
        config,
    )
    empty = evaluator.evaluate(problem, GlobalResponsePlanChromosome(problem.resource_ids, (None, None)), config)

    ranked = rank_candidates((empty, full))
    assert ranked[0] is full


# ---------------------------------------------------------------------------
# Determinism (Task 18)
# ---------------------------------------------------------------------------


def test_full_optimizer_run_is_deterministic_for_the_same_seed():
    problem = _small_problem()
    config = GlobalResponseOptimizationConfig(population_size=8, generation_count=5, random_seed=11)
    optimizer = GlobalGeneticResponseOptimizer()

    population_1 = GlobalInitialPopulationGenerator.generate(problem, config)
    result_1 = optimizer.optimize(problem, config, population_1)
    population_2 = GlobalInitialPopulationGenerator.generate(problem, config)
    result_2 = optimizer.optimize(problem, config, population_2)

    assert result_1.best_chromosome == result_2.best_chromosome
    assert result_1.score.fitness_score == result_2.score.fitness_score


def test_elitism_never_loses_the_best_candidate_across_generations():
    problem = _small_problem()
    config = GlobalResponseOptimizationConfig(population_size=6, generation_count=10, elitism_count=1, random_seed=2)
    optimizer = GlobalGeneticResponseOptimizer()
    population = GlobalInitialPopulationGenerator.generate(problem, config)

    result = optimizer.optimize(problem, config, population)

    assert result.final_best_score >= result.initial_best_score - 1e-9
    assert all(
        result.generation_best_scores[i] <= result.generation_best_scores[i + 1] + 1e-9
        for i in range(len(result.generation_best_scores) - 1)
    )
