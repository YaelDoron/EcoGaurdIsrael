"""Tests for the pure genetic response-plan optimizer."""
from __future__ import annotations

import random

import pytest

from src.calculators.response_optimization import (
    CandidateEvaluator,
    EvaluatedChromosome,
    GeneticResponsePlanOptimizer,
    ResponseOptimizationConfig,
    ResponsePlanChromosomeDecoder,
    ResponsePlanScorer,
    TournamentSelector,
    rank_candidates,
)
from src.models import (
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    PlanScoreBreakdown,
    ResponseOptimizationInput,
    ResponsePlanChromosome,
    ResponsePlanStatus,
    ResponseTargetType,
)


def target(target_id: int, order: int, priority: float = 100.0) -> OptimizationTarget:
    return OptimizationTarget(target_id, order, ResponseTargetType.ACTIVE_FIRE, priority)


def resource(resource_id: int | str) -> OptimizationResource:
    return OptimizationResource(resource_id)


def route(route_id: int, resource_id: int | str, target_id: int, eta: float, reachable: bool = True):
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=eta if reachable else None,
        distance_meters=1000.0 if reachable else None,
    )


def planning_input() -> ResponseOptimizationInput:
    return ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0), target(30, 2, 25.0)),
        resources=(resource("R1"), resource("R2"), resource("R3")),
        route_options=(
            route(100, "R1", 10, 300.0),
            route(101, "R2", 10, 600.0),
            route(102, "R1", 20, 900.0),
            route(200, "R2", 20, 200.0),
            route(300, "R3", 30, 100.0),
        ),
    )


def score(
    *,
    total: float,
    coverage: float = 0.0,
    covered_count: int = 0,
    average_eta: float | None = None,
) -> PlanScoreBreakdown:
    return PlanScoreBreakdown(
        total_score=total,
        raw_fitness=total,
        coverage_score=coverage,
        average_eta_seconds=average_eta,
        total_priority=100.0,
        covered_priority=coverage,
        covered_target_count=covered_count,
        total_target_count=3,
        uncovered_target_ids=(10, 20, 30)[covered_count:],
        status=ResponsePlanStatus.PARTIAL if covered_count else ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
    )


def evaluated(chromosome, score_value: PlanScoreBreakdown) -> EvaluatedChromosome:
    return EvaluatedChromosome(chromosome=ResponsePlanChromosome(chromosome), actions=(), score=score_value)


def assert_decodable(input_data: ResponseOptimizationInput, chromosome: ResponsePlanChromosome):
    actions = ResponsePlanChromosomeDecoder.decode(input_data, chromosome)
    assert len({action.resource_id for action in actions}) == len(actions)
    assert len({action.response_target_id for action in actions}) == len(actions)


def test_ranking_uses_score_coverage_count_eta_and_chromosome_key():
    assert rank_candidates((
        evaluated((None,), score(total=10.0)),
        evaluated(("R1",), score(total=20.0)),
    ))[0].chromosome == ResponsePlanChromosome(("R1",))

    assert rank_candidates((
        evaluated(("R1",), score(total=10.0, coverage=20.0)),
        evaluated(("R2",), score(total=10.0, coverage=30.0)),
    ))[0].chromosome == ResponsePlanChromosome(("R2",))

    assert rank_candidates((
        evaluated(("R1",), score(total=10.0, coverage=30.0, covered_count=1)),
        evaluated(("R2",), score(total=10.0, coverage=30.0, covered_count=2)),
    ))[0].chromosome == ResponsePlanChromosome(("R2",))

    assert rank_candidates((
        evaluated(("R1",), score(total=10.0, coverage=30.0, covered_count=2, average_eta=400.0)),
        evaluated(("R2",), score(total=10.0, coverage=30.0, covered_count=2, average_eta=200.0)),
        evaluated(("R3",), score(total=10.0, coverage=30.0, covered_count=2, average_eta=None)),
    ))[0].chromosome == ResponsePlanChromosome(("R2",))

    assert rank_candidates((
        evaluated(("R2",), score(total=10.0, coverage=30.0, covered_count=2, average_eta=200.0)),
        evaluated(("R1",), score(total=10.0, coverage=30.0, covered_count=2, average_eta=200.0)),
    ))[0].chromosome == ResponsePlanChromosome(("R1",))


def test_tournament_selection_is_deterministic_and_does_not_mutate_population():
    population = (
        evaluated(("R1",), score(total=10.0)),
        evaluated(("R2",), score(total=20.0)),
        evaluated(("R3",), score(total=30.0)),
    )
    config = ResponseOptimizationConfig(population_size=3, tournament_size=2)

    first = TournamentSelector.select(population, config, random.Random(5))
    second = TournamentSelector.select(population, config, random.Random(5))

    assert first == second
    assert first in population
    assert population[0].chromosome == ResponsePlanChromosome(("R1",))


def test_tournament_selection_handles_identical_candidates():
    candidate = evaluated((None,), score(total=0.0))
    population = (candidate, candidate)

    assert TournamentSelector.select(
        population,
        ResponseOptimizationConfig(population_size=2, tournament_size=2),
        random.Random(1),
    ) == candidate


def test_candidate_evaluator_uses_decoder_and_scorer():
    input_data = planning_input()
    candidate = CandidateEvaluator().evaluate(input_data, ResponsePlanChromosome(("R1", "R2", None)))
    expected = ResponsePlanScorer().evaluate(input_data, candidate.actions)

    assert candidate.score == expected
    assert [(action.resource_id, action.response_target_id) for action in candidate.actions] == [("R1", 10), ("R2", 20)]


def test_crossover_preserves_feasibility_and_inherits_only_parent_genes():
    input_data = planning_input()
    parent_a = ResponsePlanChromosome(("R1", None, "R3"))
    parent_b = ResponsePlanChromosome((None, "R1", "R2"))
    feasible = __import__(
        "src.calculators.response_optimization.feasible_route_lookup",
        fromlist=["feasible_resource_ids_by_target"],
    ).feasible_resource_ids_by_target(input_data)
    config = ResponseOptimizationConfig(population_size=4, crossover_rate=1.0)

    for seed in range(20):
        child_a, child_b = GeneticResponsePlanOptimizer._recombine(
            parent_a,
            parent_b,
            input_data,
            feasible,
            config,
            random.Random(seed),
        )
        for child in (child_a, child_b):
            assert len(child.genes) == len(parent_a.genes)
            assert len([gene for gene in child.genes if gene is not None]) == len(set(gene for gene in child.genes if gene is not None))
            for index, gene in enumerate(child.genes):
                assert gene in (parent_a.genes[index], parent_b.genes[index], None)
            assert_decodable(input_data, child)


def test_crossover_rate_zero_copies_parents_before_mutation():
    input_data = planning_input()
    feasible = {target.response_target_id: () for target in input_data.targets}
    parent_a = ResponsePlanChromosome(("R1", None, None))
    parent_b = ResponsePlanChromosome((None, "R2", None))

    assert GeneticResponsePlanOptimizer._recombine(
        parent_a,
        parent_b,
        input_data,
        feasible,
        ResponseOptimizationConfig(population_size=4, crossover_rate=0.0),
        random.Random(1),
    ) == (parent_a, parent_b)


def test_mutation_rate_zero_leaves_chromosome_unchanged():
    input_data = planning_input()
    feasible = __import__(
        "src.calculators.response_optimization.feasible_route_lookup",
        fromlist=["feasible_resource_ids_by_target"],
    ).feasible_resource_ids_by_target(input_data)
    chromosome = ResponsePlanChromosome(("R1", None, "R3"))

    assert GeneticResponsePlanOptimizer._mutate(
        chromosome,
        input_data,
        feasible,
        ResponseOptimizationConfig(population_size=4, mutation_rate=0.0),
        random.Random(1),
    ) == chromosome


def test_mutation_preserves_feasibility_and_is_seeded():
    input_data = planning_input()
    feasible = __import__(
        "src.calculators.response_optimization.feasible_route_lookup",
        fromlist=["feasible_resource_ids_by_target"],
    ).feasible_resource_ids_by_target(input_data)
    chromosome = ResponsePlanChromosome(("R1", None, "R3"))
    config = ResponseOptimizationConfig(population_size=4, mutation_rate=1.0)

    first = GeneticResponsePlanOptimizer._mutate(chromosome, input_data, feasible, config, random.Random(3))
    second = GeneticResponsePlanOptimizer._mutate(chromosome, input_data, feasible, config, random.Random(3))

    assert first == second
    assert first != chromosome
    assert_decodable(input_data, first)


def test_full_optimizer_returns_valid_deterministic_result_and_non_degrading_history():
    input_data = planning_input()
    config = ResponseOptimizationConfig(population_size=8, generation_count=5, random_seed=9, elitism_count=1)

    first = GeneticResponsePlanOptimizer().optimize(input_data, config)
    second = GeneticResponsePlanOptimizer().optimize(input_data, config)

    assert first == second
    assert_decodable(input_data, first.best_chromosome)
    assert first.score == ResponsePlanScorer(config).evaluate(input_data, first.actions)
    assert first.generations_executed == 5
    assert first.population_size == 8
    assert all(
        later + 1e-9 >= earlier
        for earlier, later in zip(first.generation_best_scores, first.generation_best_scores[1:])
    )


def test_full_optimizer_finds_known_simple_optimum():
    input_data = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(10, 0, 100.0),),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R1", 10, 100.0), route(200, "R2", 10, 1000.0)),
    )
    config = ResponseOptimizationConfig(
        population_size=4,
        generation_count=3,
        random_seed=2,
        mutation_rate=0.0,
        crossover_rate=0.0,
    )

    result = GeneticResponsePlanOptimizer().optimize(input_data, config)

    assert result.best_chromosome == ResponsePlanChromosome(("R1",))
    assert result.actions[0].route_result_id == 100


def test_full_optimizer_zero_feasibility_returns_empty_no_feasible_result():
    input_data = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R1"),),
        route_options=(route(100, "R1", 10, 0.0, reachable=False),),
    )

    result = GeneticResponsePlanOptimizer().optimize(
        input_data,
        ResponseOptimizationConfig(population_size=4, generation_count=2),
    )

    assert result.best_chromosome == ResponsePlanChromosome((None, None))
    assert result.actions == ()
    assert result.score.total_score == pytest.approx(0.0)
    assert result.score.coverage_score == pytest.approx(0.0)
    assert result.score.status is ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS


def test_equivalent_inputs_with_different_raw_order_have_identical_result():
    first = planning_input()
    second = ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=(target(30, 2, 25.0), target(10, 0, 100.0), target(20, 1, 50.0)),
        resources=(resource("R3"), resource("R1"), resource("R2")),
        route_options=(
            route(300, "R3", 30, 100.0),
            route(102, "R1", 20, 900.0),
            route(200, "R2", 20, 200.0),
            route(101, "R2", 10, 600.0),
            route(100, "R1", 10, 300.0),
        ),
    )
    config = ResponseOptimizationConfig(population_size=8, generation_count=3, random_seed=4)

    assert first == second
    assert GeneticResponsePlanOptimizer().optimize(first, config) == GeneticResponsePlanOptimizer().optimize(second, config)
