"""Pure Global GA core (Stage 4, Tasks 16-19; hard-dispatch-lock repair
added Stage 6, Task 16).

Crossover/mutation operate on the resource-indexed chromosome: resource
uniqueness is automatic (one gene per resource_id); slot uniqueness is
enforced incrementally as each child/mutated gene is chosen - a candidate
slot is only accepted if not already claimed elsewhere in that same
chromosome, which IS the repair step (Task 16), not a separate pass, and
functions identically whether the duplicate came from crossover or
mutation. Selection/elitism mirror the legacy per-event GA's tournament
approach (Task 18) - a different evolutionary framework is not warranted.

Every child/mutated chromosome is additionally run through
`enforce_hard_locks` (Stage 6) as a final repair step: crossover/mutation
may legally choose `None` for a hard-dispatched resource (their internal
candidate sets already exclude any OTHER FireEvent's slots, since
`feasible_slot_ids_by_resource` was pre-restricted at problem-build time),
but a locked resource must never actually end up idle - the repair step
re-assigns it to one of its own feasible slots, evicting a non-locked
competitor if necessary. This keeps the illegal "crossed to another event"
region of the search space structurally unreachable, and additionally
guarantees "never idle" without needing operator-level special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from src.calculators.global_response_optimization.global_hard_dispatch_lock import enforce_hard_locks
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    GlobalResponseOptimizationProblem,
)
from src.calculators.global_response_optimization.global_response_plan_chromosome_decoder import (
    GlobalResponsePlanChromosomeDecoder,
)
from src.calculators.global_response_optimization.global_response_plan_scorer import (
    GlobalResponsePlanScorer,
    GlobalScoreBreakdown,
)
from src.models.global_response_action import GlobalResponseAction
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome


@dataclass(frozen=True)
class GlobalEvaluatedChromosome:
    """One decoded and scored chromosome for a single generation."""

    chromosome: GlobalResponsePlanChromosome
    actions: tuple[GlobalResponseAction, ...]
    score: GlobalScoreBreakdown


@dataclass(frozen=True)
class GlobalGeneticOptimizationResult:
    """Pure GA output consumed by GlobalResponseOptimizationService."""

    best_chromosome: GlobalResponsePlanChromosome
    actions: tuple[GlobalResponseAction, ...]
    score: GlobalScoreBreakdown
    generations_executed: int
    population_size: int
    initial_best_score: float
    final_best_score: float
    generation_best_scores: tuple[float, ...]


class GlobalCandidateEvaluator:
    """Decode + score chromosomes; reusable across runs since config is passed per call."""

    def __init__(self, scorer: GlobalResponsePlanScorer | None = None) -> None:
        self._scorer = scorer

    def evaluate(
        self,
        problem: GlobalResponseOptimizationProblem,
        chromosome: GlobalResponsePlanChromosome,
        config: GlobalResponseOptimizationConfig,
    ) -> GlobalEvaluatedChromosome:
        actions = GlobalResponsePlanChromosomeDecoder.decode(problem, chromosome)
        scorer = self._scorer or GlobalResponsePlanScorer(config)
        score = scorer.evaluate(problem, chromosome)
        return GlobalEvaluatedChromosome(chromosome=chromosome, actions=actions, score=score)


def chromosome_key(chromosome: GlobalResponsePlanChromosome) -> tuple[tuple[int, str], ...]:
    """Deterministic gene key: (0, "") for idle, (1, slot_id) otherwise, per resource position."""
    return tuple((0, "") if gene is None else (1, gene) for gene in chromosome.genes)


def candidate_ranking_key(candidate: GlobalEvaluatedChromosome):
    """Task 19's exact tie-break: fitness desc, covered-slot-count desc, total ETA asc, chromosome key."""
    total_eta = math.inf if candidate.score.average_eta_seconds is None else (
        candidate.score.average_eta_seconds * candidate.score.covered_slot_count
    )
    return (
        -candidate.score.fitness_score,
        -candidate.score.covered_slot_count,
        total_eta,
        chromosome_key(candidate.chromosome),
    )


def rank_candidates(candidates: tuple[GlobalEvaluatedChromosome, ...]) -> tuple[GlobalEvaluatedChromosome, ...]:
    return tuple(sorted(candidates, key=candidate_ranking_key))


class GlobalTournamentSelector:
    """Seeded tournament selection over an evaluated population."""

    @staticmethod
    def select(
        evaluated_population: tuple[GlobalEvaluatedChromosome, ...],
        config: GlobalResponseOptimizationConfig,
        rng: random.Random,
    ) -> GlobalEvaluatedChromosome:
        if len(evaluated_population) < config.tournament_size:
            raise ValueError("evaluated population is smaller than tournament_size.")
        contestants = rng.sample(list(evaluated_population), config.tournament_size)
        return rank_candidates(tuple(contestants))[0]


class GlobalGeneticResponseOptimizer:
    """Pure GA over feasible resource-indexed chromosomes."""

    def __init__(self, evaluator: GlobalCandidateEvaluator | None = None) -> None:
        self._evaluator = evaluator or GlobalCandidateEvaluator()

    def optimize(
        self,
        problem: GlobalResponseOptimizationProblem,
        config: GlobalResponseOptimizationConfig,
        initial_population: tuple[GlobalResponsePlanChromosome, ...],
    ) -> GlobalGeneticOptimizationResult:
        if not isinstance(problem, GlobalResponseOptimizationProblem):
            raise ValueError(f"problem must be a GlobalResponseOptimizationProblem, got {problem!r}")
        if not isinstance(config, GlobalResponseOptimizationConfig):
            raise ValueError(f"config must be a GlobalResponseOptimizationConfig, got {config!r}")

        rng = random.Random(config.random_seed)
        population = tuple(initial_population)
        evaluated = self._evaluate_population(problem, population, config)
        ranked = rank_candidates(evaluated)
        generation_best_scores = [ranked[0].score.fitness_score]
        initial_best_score = ranked[0].score.fitness_score

        for _ in range(config.generation_count):
            ranked = rank_candidates(evaluated)
            next_population = [candidate.chromosome for candidate in ranked[: config.elitism_count]]

            while len(next_population) < config.population_size:
                parent_a = GlobalTournamentSelector.select(evaluated, config, rng).chromosome
                parent_b = GlobalTournamentSelector.select(evaluated, config, rng).chromosome
                child_a, child_b = self._recombine(parent_a, parent_b, problem, config, rng)
                child_a = self._mutate(child_a, problem, config, rng)
                child_b = self._mutate(child_b, problem, config, rng)
                next_population.append(child_a)
                if len(next_population) < config.population_size:
                    next_population.append(child_b)

            population = tuple(next_population)
            evaluated = self._evaluate_population(problem, population, config)
            generation_best_scores.append(rank_candidates(evaluated)[0].score.fitness_score)

        best = rank_candidates(evaluated)[0]
        return GlobalGeneticOptimizationResult(
            best_chromosome=best.chromosome,
            actions=best.actions,
            score=best.score,
            generations_executed=config.generation_count,
            population_size=config.population_size,
            initial_best_score=initial_best_score,
            final_best_score=best.score.fitness_score,
            generation_best_scores=tuple(generation_best_scores),
        )

    def _evaluate_population(
        self,
        problem: GlobalResponseOptimizationProblem,
        population: tuple[GlobalResponsePlanChromosome, ...],
        config: GlobalResponseOptimizationConfig,
    ) -> tuple[GlobalEvaluatedChromosome, ...]:
        return tuple(self._evaluator.evaluate(problem, chromosome, config) for chromosome in population)

    @classmethod
    def _recombine(
        cls,
        parent_a: GlobalResponsePlanChromosome,
        parent_b: GlobalResponsePlanChromosome,
        problem: GlobalResponseOptimizationProblem,
        config: GlobalResponseOptimizationConfig,
        rng: random.Random,
    ) -> tuple[GlobalResponsePlanChromosome, GlobalResponsePlanChromosome]:
        if rng.random() >= config.crossover_rate:
            return parent_a, parent_b
        return (
            cls._uniform_child(parent_a, parent_b, problem, rng),
            cls._uniform_child(parent_b, parent_a, problem, rng),
        )

    @staticmethod
    def _uniform_child(
        parent_a: GlobalResponsePlanChromosome,
        parent_b: GlobalResponsePlanChromosome,
        problem: GlobalResponseOptimizationProblem,
        rng: random.Random,
    ) -> GlobalResponsePlanChromosome:
        used_slots: set[str] = set()
        genes: list[str | None] = []
        for index, resource_id in enumerate(problem.resource_ids):
            first, second = parent_a.genes[index], parent_b.genes[index]
            candidates = (first, second, None) if rng.random() < 0.5 else (second, first, None)
            feasible_slots = problem.feasible_slot_ids_by_resource[resource_id]
            selected = None
            for candidate in candidates:
                if candidate is None:
                    selected = None
                    break
                if candidate not in used_slots and candidate in feasible_slots:
                    selected = candidate
                    used_slots.add(candidate)
                    break
            genes.append(selected)
        return enforce_hard_locks(problem, GlobalResponsePlanChromosome(problem.resource_ids, tuple(genes)))

    @staticmethod
    def _mutate(
        chromosome: GlobalResponsePlanChromosome,
        problem: GlobalResponseOptimizationProblem,
        config: GlobalResponseOptimizationConfig,
        rng: random.Random,
    ) -> GlobalResponsePlanChromosome:
        genes = list(chromosome.genes)
        used_slots = {gene for gene in genes if gene is not None}

        for index, resource_id in enumerate(problem.resource_ids):
            if rng.random() >= config.mutation_rate:
                continue
            current = genes[index]
            if current is not None:
                used_slots.discard(current)

            feasible_unused = [
                slot_id
                for slot_id in problem.feasible_slot_ids_by_resource[resource_id]
                if slot_id not in used_slots
            ]
            choices: list[str | None] = [None, *feasible_unused]
            alternatives = [choice for choice in choices if choice != current]
            selected = rng.choice(alternatives or choices)
            genes[index] = selected
            if selected is not None:
                used_slots.add(selected)

        return enforce_hard_locks(problem, GlobalResponsePlanChromosome(problem.resource_ids, tuple(genes)))
