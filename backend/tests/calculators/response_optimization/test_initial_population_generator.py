"""Tests for deterministic initial response-plan populations."""
from __future__ import annotations

from src.calculators.response_optimization import (
    InitialPopulationGenerator,
    ResponseOptimizationConfig,
    ResponsePlanChromosomeDecoder,
)
from src.models import (
    OptimizationResource,
    OptimizationRouteOption,
    OptimizationTarget,
    ResponseOptimizationInput,
    ResponsePlanChromosome,
    ResponseTargetType,
)


def target(target_id: int, order: int, priority: float = 100.0) -> OptimizationTarget:
    return OptimizationTarget(target_id, order, ResponseTargetType.ACTIVE_FIRE, priority)


def resource(resource_id: int | str) -> OptimizationResource:
    return OptimizationResource(resource_id)


def route(
    route_id: int,
    resource_id: int | str,
    target_id: int,
    *,
    reachable: bool = True,
) -> OptimizationRouteOption:
    return OptimizationRouteOption(
        route_result_id=route_id,
        resource_id=resource_id,
        response_target_id=target_id,
        is_reachable=reachable,
        travel_time_seconds=120.0 if reachable else None,
        distance_meters=800.0 if reachable else None,
    )


def input_data(
    *,
    targets=(target(10, 0), target(20, 1), target(30, 2)),
    resources=(resource("R1"), resource("R2"), resource("R3")),
    route_options=(
        route(100, "R1", 10),
        route(101, "R2", 10),
        route(200, "R2", 20),
        route(300, "R3", 30),
    ),
) -> ResponseOptimizationInput:
    return ResponseOptimizationInput(
        fire_event_id=1,
        response_target_set_id=2,
        route_planning_run_id=3,
        targets=targets,
        resources=resources,
        route_options=route_options,
    )


def generate(
    optimization_input: ResponseOptimizationInput,
    *,
    size: int = 8,
    seed: int = 42,
    probability: float = 0.75,
) -> tuple[ResponsePlanChromosome, ...]:
    return InitialPopulationGenerator.generate(
        optimization_input,
        ResponseOptimizationConfig(
            population_size=size,
            random_seed=seed,
            initial_assignment_probability=probability,
        ),
    )


def assert_every_chromosome_decodes(input_value: ResponseOptimizationInput, population):
    for chromosome in population:
        actions = ResponsePlanChromosomeDecoder.decode(input_value, chromosome)
        assert len({action.resource_id for action in actions}) == len(actions)


def test_population_size_seed_determinism_and_required_seeds():
    data = input_data()
    population = generate(data, size=6, seed=7)

    assert len(population) == 6
    assert population == generate(data, size=6, seed=7)
    assert population[0] == ResponsePlanChromosome((None, None, None))
    assert population[1] == ResponsePlanChromosome(("R1", "R2", "R3"))
    assert_every_chromosome_decodes(data, population)


def test_different_seeds_can_produce_different_random_members():
    data = input_data()

    assert generate(data, size=8, seed=1)[2:] != generate(data, size=8, seed=2)[2:]


def test_zero_resources_returns_requested_number_of_empty_chromosomes():
    data = input_data(resources=(), route_options=())

    assert generate(data, size=5) == (ResponsePlanChromosome((None, None, None)),) * 5


def test_all_routes_unreachable_returns_empty_chromosomes():
    data = input_data(
        route_options=(
            route(100, "R1", 10, reachable=False),
            route(200, "R2", 20, reachable=False),
            route(300, "R3", 30, reachable=False),
        )
    )

    assert generate(data, size=4) == (ResponsePlanChromosome((None, None, None)),) * 4


def test_partially_reachable_graph_uses_only_reachable_pairs():
    data = input_data(
        route_options=(
            route(100, "R1", 10),
            route(200, "R2", 20, reachable=False),
            route(300, "R3", 30),
        )
    )

    population = generate(data, size=8, probability=1.0)

    assert_every_chromosome_decodes(data, population)
    for chromosome in population:
        assert chromosome.genes[1] is None


def test_more_targets_than_resources_and_more_resources_than_targets_work():
    more_targets = input_data(
        targets=(target(10, 0), target(20, 1), target(30, 2), target(40, 3)),
        resources=(resource("R1"), resource("R2")),
        route_options=(route(100, "R1", 10), route(200, "R2", 20)),
    )
    more_resources = input_data(
        targets=(target(10, 0), target(20, 1)),
        resources=(resource("R1"), resource("R2"), resource("R3")),
        route_options=(route(100, "R1", 10), route(200, "R2", 20), route(201, "R3", 20)),
    )

    assert_every_chromosome_decodes(more_targets, generate(more_targets, size=6))
    assert_every_chromosome_decodes(more_resources, generate(more_resources, size=6))


def test_exact_population_size_when_unique_space_is_smaller():
    data = input_data(targets=(target(10, 0),), resources=(), route_options=())

    population = generate(data, size=7)

    assert len(population) == 7
    assert population == (ResponsePlanChromosome((None,)),) * 7


def test_assignment_probability_boundaries_for_random_candidates():
    data = input_data()

    zero = generate(data, size=5, probability=0.0)
    one = generate(data, size=5, probability=1.0)

    assert zero[2:] == (ResponsePlanChromosome((None, None, None)),) * 3
    assert all(any(gene is not None for gene in chromosome.genes) for chromosome in one[1:])
    assert_every_chromosome_decodes(data, one)


def test_logically_equivalent_inputs_generate_same_population_after_normalization():
    canonical = input_data()
    shuffled = input_data(
        targets=(target(30, 2), target(10, 0), target(20, 1)),
        resources=(resource("R3"), resource("R1"), resource("R2")),
        route_options=(
            route(300, "R3", 30),
            route(101, "R2", 10),
            route(200, "R2", 20),
            route(100, "R1", 10),
        ),
    )

    assert canonical == shuffled
    assert generate(canonical, seed=11) == generate(shuffled, seed=11)
