"""Deterministic initial population for the Global GA (Stage 4 Task 12;
demand-aware value function since Stage 5; hard-dispatch-lock repair since
Stage 6).

Seeds:
  1. all-idle EXCEPT locked resources, which are never idle (Stage 6) -
     every OTHER resource unassigned;
  2. greedy global demand-aware value - considers every feasible
     (resource, slot) pair TOGETHER (not per FireEvent), sorted by
     descending GlobalResponsePlanScorer.slot_value (REQUIRED > DESIRED >
     PREDICTED_RISK, severity/ETA breaking ties within a tier - Stage 5),
     greedily claiming pairs while respecting one-resource/one-slot;
  3..population_size: random feasible chromosomes controlled by
     random_seed/initial_assignment_probability.

Every seed is run through `enforce_hard_locks` (Stage 6, Task 16) as a
final repair step, so no population member ever leaves a hard-dispatched
resource unassigned or assigned outside its own FireEvent.
"""
from __future__ import annotations

import random

from src.calculators.global_response_optimization.global_hard_dispatch_lock import enforce_hard_locks
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    GlobalResponseOptimizationProblem,
)
from src.calculators.global_response_optimization.global_response_plan_scorer import GlobalResponsePlanScorer
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome


class GlobalInitialPopulationGenerator:
    """Create feasible resource-indexed chromosomes without scoring them (beyond the greedy seed's own ranking)."""

    @classmethod
    def generate(
        cls,
        problem: GlobalResponseOptimizationProblem,
        config: GlobalResponseOptimizationConfig,
        rng: random.Random | None = None,
        scorer: GlobalResponsePlanScorer | None = None,
    ) -> tuple[GlobalResponsePlanChromosome, ...]:
        if not isinstance(problem, GlobalResponseOptimizationProblem):
            raise ValueError(f"problem must be a GlobalResponseOptimizationProblem, got {problem!r}")
        if not isinstance(config, GlobalResponseOptimizationConfig):
            raise ValueError(f"config must be a GlobalResponseOptimizationConfig, got {config!r}")

        scorer = scorer or GlobalResponsePlanScorer(config)
        empty = GlobalResponsePlanChromosome(problem.resource_ids, (None,) * len(problem.resource_ids))
        greedy_seed = cls._greedy_seed(problem, scorer)

        population = [enforce_hard_locks(problem, empty)]
        if len(population) < config.population_size:
            population.append(enforce_hard_locks(problem, greedy_seed))

        rng = rng or random.Random(config.random_seed)
        while len(population) < config.population_size:
            population.append(enforce_hard_locks(problem, cls._random_candidate(problem, config, rng)))

        return tuple(population)

    @staticmethod
    def _greedy_seed(
        problem: GlobalResponseOptimizationProblem, scorer: GlobalResponsePlanScorer
    ) -> GlobalResponsePlanChromosome:
        candidates: list[tuple[float, str, str]] = []
        for resource_id in problem.resource_ids:
            for slot_id in problem.feasible_slot_ids_by_resource[resource_id]:
                route = problem.route_for(resource_id, slot_id)
                slot = problem.slots_by_id[slot_id]
                value = scorer.slot_value(problem, slot, route.eta_seconds, resource_id)
                candidates.append((value, resource_id, slot_id))
        # Highest value first; resource_id/slot_id break ties deterministically.
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))

        used_resources: set[str] = set()
        used_slots: set[str] = set()
        gene_by_resource: dict[str, str] = {}
        for _value, resource_id, slot_id in candidates:
            if resource_id in used_resources or slot_id in used_slots:
                continue
            gene_by_resource[resource_id] = slot_id
            used_resources.add(resource_id)
            used_slots.add(slot_id)

        genes = tuple(gene_by_resource.get(resource_id) for resource_id in problem.resource_ids)
        return GlobalResponsePlanChromosome(problem.resource_ids, genes)

    @staticmethod
    def _random_candidate(
        problem: GlobalResponseOptimizationProblem,
        config: GlobalResponseOptimizationConfig,
        rng: random.Random,
    ) -> GlobalResponsePlanChromosome:
        genes: list[str | None] = [None] * len(problem.resource_ids)
        used_slots: set[str] = set()
        resource_indices = list(range(len(problem.resource_ids)))
        rng.shuffle(resource_indices)

        for index in resource_indices:
            resource_id = problem.resource_ids[index]
            feasible_unused = [
                slot_id
                for slot_id in problem.feasible_slot_ids_by_resource[resource_id]
                if slot_id not in used_slots
            ]
            if not feasible_unused:
                continue
            if rng.random() > config.initial_assignment_probability:
                continue
            slot_id = rng.choice(feasible_unused)
            genes[index] = slot_id
            used_slots.add(slot_id)

        return GlobalResponsePlanChromosome(problem.resource_ids, tuple(genes))
