"""Deterministic initial population generation for response-plan chromosomes."""
from __future__ import annotations

import random

from src.calculators.response_optimization.feasible_route_lookup import feasible_resource_ids_by_target
from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.response_plan_chromosome import ResponsePlanChromosome


class InitialPopulationGenerator:
    """Create feasible target-indexed chromosomes without scoring them."""

    # Internal search-budget knob, not part of the deterministic config
    # contract: it caps how many extra draws are spent trying to avoid
    # duplicate chromosomes while filling out the initial population before
    # falling back to allowing duplicates. Changing it does shift the shared
    # rng stream (and, when duplicates are unavoidable, which specific
    # "long-tail" duplicate chromosomes end up in the population) - but it
    # was empirically verified (5 vs 20, 200+ seeds, including scenarios
    # engineered to force heavy duplicate-retry exhaustion via a small
    # feasible-chromosome space relative to population_size) to never change
    # the ranked winner of the initial population or the final GA output.
    # The deterministic empty/coverage seeds anchor the population
    # regardless, and elitism/selection converge to the same optimum
    # independent of which duplicate-filler chromosomes fill the remaining
    # slots. It is therefore kept as an internal constant rather than a
    # ResponseOptimizationConfig field.
    _DUPLICATE_RETRY_MULTIPLIER = 20

    @classmethod
    def generate(
        cls,
        optimization_input: ResponseOptimizationInput,
        config: ResponseOptimizationConfig,
        rng: random.Random | None = None,
    ) -> tuple[ResponsePlanChromosome, ...]:
        if not isinstance(optimization_input, ResponseOptimizationInput):
            raise ValueError(f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}")
        if not isinstance(config, ResponseOptimizationConfig):
            raise ValueError(f"config must be a ResponseOptimizationConfig, got {config!r}")

        target_count = len(optimization_input.targets)
        empty = ResponsePlanChromosome((None,) * target_count)
        feasible_by_target = feasible_resource_ids_by_target(optimization_input)
        coverage_seed = cls._coverage_oriented_seed(optimization_input, feasible_by_target)

        population = [empty]
        if len(population) < config.population_size:
            population.append(coverage_seed)

        seen = {empty, coverage_seed}
        rng = rng or random.Random(config.random_seed)
        retry_limit = max(config.population_size * cls._DUPLICATE_RETRY_MULTIPLIER, 1)
        retries = 0

        while len(population) < config.population_size and retries < retry_limit:
            candidate = cls._random_candidate(optimization_input, feasible_by_target, config, rng)
            if candidate in seen:
                retries += 1
                continue
            population.append(candidate)
            seen.add(candidate)
            retries = 0

        while len(population) < config.population_size:
            population.append(cls._random_candidate(optimization_input, feasible_by_target, config, rng))

        return tuple(population)

    @staticmethod
    def _coverage_oriented_seed(
        optimization_input: ResponseOptimizationInput,
        feasible_by_target: dict[int, tuple[int | str, ...]],
    ) -> ResponsePlanChromosome:
        """Build a deterministic coverage seed; this is not the baseline plan."""
        genes: list[int | str | None] = [None] * len(optimization_input.targets)
        available_resources = {resource.resource_id for resource in optimization_input.resources}
        for index, target in enumerate(optimization_input.targets):
            for resource_id in feasible_by_target[target.response_target_id]:
                if resource_id in available_resources:
                    genes[index] = resource_id
                    available_resources.remove(resource_id)
                    break
        return ResponsePlanChromosome(tuple(genes))

    @staticmethod
    def _random_candidate(
        optimization_input: ResponseOptimizationInput,
        feasible_by_target: dict[int, tuple[int | str, ...]],
        config: ResponseOptimizationConfig,
        rng: random.Random,
    ) -> ResponsePlanChromosome:
        genes: list[int | str | None] = [None] * len(optimization_input.targets)
        available_resources = {resource.resource_id for resource in optimization_input.resources}
        target_indices = list(range(len(optimization_input.targets)))
        rng.shuffle(target_indices)

        for index in target_indices:
            target = optimization_input.targets[index]
            reachable_resources = [
                resource_id
                for resource_id in feasible_by_target[target.response_target_id]
                if resource_id in available_resources
            ]
            if not reachable_resources:
                continue
            if rng.random() > config.initial_assignment_probability:
                continue
            resource_id = rng.choice(reachable_resources)
            genes[index] = resource_id
            available_resources.remove(resource_id)

        return ResponsePlanChromosome(tuple(genes))
