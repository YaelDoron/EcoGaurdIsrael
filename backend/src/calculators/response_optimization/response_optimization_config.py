"""Configuration for future response-plan genetic optimization."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

METHODOLOGY = "GENETIC_RESOURCE_ALLOCATION"
METHODOLOGY_VERSION = "1.0"

DEFAULT_POPULATION_SIZE = 24
DEFAULT_GENERATION_COUNT = 40
DEFAULT_MUTATION_RATE = 0.08
DEFAULT_CROSSOVER_RATE = 0.75
DEFAULT_RANDOM_SEED = 42
DEFAULT_INITIAL_ASSIGNMENT_PROBABILITY = 0.75
DEFAULT_TOURNAMENT_SIZE = 2
DEFAULT_ELITISM_COUNT = 1
ETA_REFERENCE_SECONDS = 900.0


@dataclass(frozen=True)
class ResponseOptimizationConfig:
    """Tunable GA configuration without a fitness formula."""

    population_size: int = DEFAULT_POPULATION_SIZE
    generation_count: int = DEFAULT_GENERATION_COUNT
    mutation_rate: float = DEFAULT_MUTATION_RATE
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    random_seed: int = DEFAULT_RANDOM_SEED
    eta_reference_seconds: float = ETA_REFERENCE_SECONDS
    initial_assignment_probability: float = DEFAULT_INITIAL_ASSIGNMENT_PROBABILITY
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE
    elitism_count: int = DEFAULT_ELITISM_COUNT

    def __post_init__(self) -> None:
        if isinstance(self.population_size, bool) or not isinstance(self.population_size, int) or self.population_size < 2:
            raise ValueError(f"population_size must be an integer >= 2, got {self.population_size!r}")
        if isinstance(self.generation_count, bool) or not isinstance(self.generation_count, int) or self.generation_count < 1:
            raise ValueError(f"generation_count must be an integer >= 1, got {self.generation_count!r}")
        self._validate_rate("mutation_rate", self.mutation_rate)
        self._validate_rate("crossover_rate", self.crossover_rate)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError(f"random_seed must be an integer, got {self.random_seed!r}")
        self._validate_positive_finite("eta_reference_seconds", self.eta_reference_seconds)
        self._validate_rate("initial_assignment_probability", self.initial_assignment_probability)
        if (
            isinstance(self.tournament_size, bool)
            or not isinstance(self.tournament_size, int)
            or self.tournament_size < 2
            or self.tournament_size > self.population_size
        ):
            raise ValueError(
                "tournament_size must be an integer between 2 and population_size, "
                f"got {self.tournament_size!r}"
            )
        if (
            isinstance(self.elitism_count, bool)
            or not isinstance(self.elitism_count, int)
            or self.elitism_count < 0
            or self.elitism_count >= self.population_size
        ):
            raise ValueError(
                "elitism_count must be an integer >= 0 and < population_size, "
                f"got {self.elitism_count!r}"
            )

    @staticmethod
    def _validate_rate(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError(f"{field_name} must be within [0, 1], got {value!r}")

    @staticmethod
    def _validate_positive_finite(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be a finite positive number, got {value!r}")
