"""Pure genetic optimization core for response-plan chromosomes."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from src.calculators.response_optimization.chromosome_decoder import ResponsePlanChromosomeDecoder
from src.calculators.response_optimization.feasible_route_lookup import feasible_resource_ids_by_target
from src.calculators.response_optimization.initial_population_generator import InitialPopulationGenerator
from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer
from src.models.optimization_validation import resource_sort_key
from src.models.plan_score_breakdown import PlanScoreBreakdown
from src.models.response_action import ResponseAction
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.response_plan_chromosome import ResponsePlanChromosome


@dataclass(frozen=True)
class EvaluatedChromosome:
    """One decoded and scored chromosome for a single generation."""

    chromosome: ResponsePlanChromosome
    actions: tuple[ResponseAction, ...]
    score: PlanScoreBreakdown


@dataclass(frozen=True)
class GeneticOptimizationResult:
    """Pure GA output consumed by later response-plan construction."""

    best_chromosome: ResponsePlanChromosome
    actions: tuple[ResponseAction, ...]
    score: PlanScoreBreakdown
    generations_executed: int
    population_size: int
    initial_best_score: float
    final_best_score: float
    generation_best_scores: tuple[float, ...]


class CandidateEvaluator:
    """Decode chromosomes through Task 3 and score them through Task 2."""

    def __init__(self, scorer: ResponsePlanScorer | None = None) -> None:
        self._scorer = scorer or ResponsePlanScorer()

    def evaluate(
        self,
        optimization_input: ResponseOptimizationInput,
        chromosome: ResponsePlanChromosome,
    ) -> EvaluatedChromosome:
        actions = ResponsePlanChromosomeDecoder.decode(optimization_input, chromosome)
        score = self._scorer.evaluate(optimization_input, actions)
        return EvaluatedChromosome(chromosome=chromosome, actions=actions, score=score)


def chromosome_key(chromosome: ResponsePlanChromosome) -> tuple[tuple[int, str, str], ...]:
    """Return a deterministic mixed-resource gene key."""
    genes = []
    for gene in chromosome.genes:
        if gene is None:
            genes.append((0, "", ""))
        else:
            type_name, value = resource_sort_key(gene)
            genes.append((1, type_name, value))
    return tuple(genes)


def candidate_ranking_key(candidate: EvaluatedChromosome):
    """Central ranking key; lower tuple means better candidate."""
    average_eta = candidate.score.average_eta_seconds
    eta_key = math.inf if average_eta is None else average_eta
    return (
        -candidate.score.total_score,
        -candidate.score.coverage_score,
        -candidate.score.covered_target_count,
        eta_key,
        chromosome_key(candidate.chromosome),
    )


def rank_candidates(candidates: tuple[EvaluatedChromosome, ...]) -> tuple[EvaluatedChromosome, ...]:
    return tuple(sorted(candidates, key=candidate_ranking_key))


class TournamentSelector:
    """Seeded tournament selection over an evaluated population."""

    @staticmethod
    def select(
        evaluated_population: tuple[EvaluatedChromosome, ...],
        config: ResponseOptimizationConfig,
        rng: random.Random,
    ) -> EvaluatedChromosome:
        if len(evaluated_population) < config.tournament_size:
            raise ValueError("evaluated population is smaller than tournament_size.")
        contestants = rng.sample(list(evaluated_population), config.tournament_size)
        return rank_candidates(tuple(contestants))[0]


class GeneticResponsePlanOptimizer:
    """Pure GA over feasible response-plan chromosomes.

    `generation_count` is the number of evolution transitions after the
    initial population. `mutation_rate` is a per-gene mutation probability.
    """

    def __init__(self, evaluator: CandidateEvaluator | None = None) -> None:
        self._evaluator = evaluator or CandidateEvaluator()

    def optimize(
        self,
        optimization_input: ResponseOptimizationInput,
        config: ResponseOptimizationConfig,
    ) -> GeneticOptimizationResult:
        if not isinstance(optimization_input, ResponseOptimizationInput):
            raise ValueError(f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}")
        if not isinstance(config, ResponseOptimizationConfig):
            raise ValueError(f"config must be a ResponseOptimizationConfig, got {config!r}")

        rng = random.Random(config.random_seed)
        population = InitialPopulationGenerator.generate(optimization_input, config, rng=rng)
        evaluated = self._evaluate_population(optimization_input, population)
        ranked = rank_candidates(evaluated)
        generation_best_scores = [ranked[0].score.total_score]
        initial_best_score = ranked[0].score.total_score
        feasible_by_target = feasible_resource_ids_by_target(optimization_input)

        for _ in range(config.generation_count):
            ranked = rank_candidates(evaluated)
            next_population = [candidate.chromosome for candidate in ranked[: config.elitism_count]]

            while len(next_population) < config.population_size:
                parent_a = TournamentSelector.select(evaluated, config, rng).chromosome
                parent_b = TournamentSelector.select(evaluated, config, rng).chromosome
                child_a, child_b = self._recombine(parent_a, parent_b, optimization_input, feasible_by_target, config, rng)
                child_a = self._mutate(child_a, optimization_input, feasible_by_target, config, rng)
                child_b = self._mutate(child_b, optimization_input, feasible_by_target, config, rng)
                next_population.append(child_a)
                if len(next_population) < config.population_size:
                    next_population.append(child_b)

            population = tuple(next_population)
            evaluated = self._evaluate_population(optimization_input, population)
            generation_best_scores.append(rank_candidates(evaluated)[0].score.total_score)

        best = rank_candidates(evaluated)[0]
        return GeneticOptimizationResult(
            best_chromosome=best.chromosome,
            actions=best.actions,
            score=best.score,
            generations_executed=config.generation_count,
            population_size=config.population_size,
            initial_best_score=initial_best_score,
            final_best_score=best.score.total_score,
            generation_best_scores=tuple(generation_best_scores),
        )

    def _evaluate_population(
        self,
        optimization_input: ResponseOptimizationInput,
        population: tuple[ResponsePlanChromosome, ...],
    ) -> tuple[EvaluatedChromosome, ...]:
        return tuple(self._evaluator.evaluate(optimization_input, chromosome) for chromosome in population)

    @classmethod
    def _recombine(
        cls,
        parent_a: ResponsePlanChromosome,
        parent_b: ResponsePlanChromosome,
        optimization_input: ResponseOptimizationInput,
        feasible_by_target: dict[int, tuple[int | str, ...]],
        config: ResponseOptimizationConfig,
        rng: random.Random,
    ) -> tuple[ResponsePlanChromosome, ResponsePlanChromosome]:
        if rng.random() >= config.crossover_rate:
            return parent_a, parent_b
        return (
            cls._uniform_child(parent_a, parent_b, optimization_input, feasible_by_target, rng),
            cls._uniform_child(parent_b, parent_a, optimization_input, feasible_by_target, rng),
        )

    @staticmethod
    def _uniform_child(
        parent_a: ResponsePlanChromosome,
        parent_b: ResponsePlanChromosome,
        optimization_input: ResponseOptimizationInput,
        feasible_by_target: dict[int, tuple[int | str, ...]],
        rng: random.Random,
    ) -> ResponsePlanChromosome:
        used_resources: set[int | str] = set()
        genes: list[int | str | None] = []
        for index, target in enumerate(optimization_input.targets):
            first, second = parent_a.genes[index], parent_b.genes[index]
            candidates = (first, second, None) if rng.random() < 0.5 else (second, first, None)
            selected = None
            feasible_resources = feasible_by_target[target.response_target_id]
            for candidate in candidates:
                if candidate is None:
                    selected = None
                    break
                if candidate not in used_resources and candidate in feasible_resources:
                    selected = candidate
                    used_resources.add(candidate)
                    break
            genes.append(selected)
        return ResponsePlanChromosome(tuple(genes))

    @staticmethod
    def _mutate(
        chromosome: ResponsePlanChromosome,
        optimization_input: ResponseOptimizationInput,
        feasible_by_target: dict[int, tuple[int | str, ...]],
        config: ResponseOptimizationConfig,
        rng: random.Random,
    ) -> ResponsePlanChromosome:
        genes = list(chromosome.genes)
        used_resources = {gene for gene in genes if gene is not None}

        for index, target in enumerate(optimization_input.targets):
            if rng.random() >= config.mutation_rate:
                continue
            current = genes[index]
            if current is not None:
                used_resources.remove(current)

            feasible_unused = [
                resource_id
                for resource_id in feasible_by_target[target.response_target_id]
                if resource_id not in used_resources
            ]
            choices: list[int | str | None] = [None, *feasible_unused]
            alternatives = [choice for choice in choices if choice != current]
            selected = rng.choice(alternatives or choices)
            genes[index] = selected
            if selected is not None:
                used_resources.add(selected)

        return ResponsePlanChromosome(tuple(genes))
