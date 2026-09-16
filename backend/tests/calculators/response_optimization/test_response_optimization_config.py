"""Tests for response-optimization configuration."""
from __future__ import annotations

import pytest

from src.calculators.response_optimization import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_GENERATION_COUNT,
    DEFAULT_INITIAL_ASSIGNMENT_PROBABILITY,
    DEFAULT_ELITISM_COUNT,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOURNAMENT_SIZE,
    ETA_REFERENCE_SECONDS,
    METHODOLOGY,
    METHODOLOGY_VERSION,
    ResponseOptimizationConfig,
)


def test_default_config_valid_and_methodology_identity():
    config = ResponseOptimizationConfig()

    assert METHODOLOGY == "GENETIC_RESOURCE_ALLOCATION"
    assert METHODOLOGY_VERSION == "1.0"
    assert config.population_size == DEFAULT_POPULATION_SIZE
    assert config.generation_count == DEFAULT_GENERATION_COUNT
    assert config.mutation_rate == DEFAULT_MUTATION_RATE
    assert config.crossover_rate == DEFAULT_CROSSOVER_RATE
    assert config.random_seed == DEFAULT_RANDOM_SEED
    assert config.eta_reference_seconds == ETA_REFERENCE_SECONDS
    assert config.initial_assignment_probability == DEFAULT_INITIAL_ASSIGNMENT_PROBABILITY
    assert config.tournament_size == DEFAULT_TOURNAMENT_SIZE
    assert config.elitism_count == DEFAULT_ELITISM_COUNT


@pytest.mark.parametrize("population_size", [0, 1, True])
def test_invalid_population_size_rejected(population_size):
    with pytest.raises(ValueError):
        ResponseOptimizationConfig(population_size=population_size)


@pytest.mark.parametrize("generation_count", [0, -1, True])
def test_invalid_generation_count_rejected(generation_count):
    with pytest.raises(ValueError):
        ResponseOptimizationConfig(generation_count=generation_count)


@pytest.mark.parametrize("mutation_rate", [0.0, 1.0])
def test_mutation_rate_boundaries_allowed(mutation_rate):
    assert ResponseOptimizationConfig(mutation_rate=mutation_rate).mutation_rate == mutation_rate


@pytest.mark.parametrize("mutation_rate", [-0.01, 1.01, True])
def test_invalid_mutation_rate_rejected(mutation_rate):
    with pytest.raises(ValueError):
        ResponseOptimizationConfig(mutation_rate=mutation_rate)


@pytest.mark.parametrize("crossover_rate", [0.0, 1.0])
def test_crossover_rate_boundaries_allowed(crossover_rate):
    assert ResponseOptimizationConfig(crossover_rate=crossover_rate).crossover_rate == crossover_rate


@pytest.mark.parametrize("crossover_rate", [-0.01, 1.01, True])
def test_invalid_crossover_rate_rejected(crossover_rate):
    with pytest.raises(ValueError):
        ResponseOptimizationConfig(crossover_rate=crossover_rate)


def test_seed_stored_deterministically():
    assert ResponseOptimizationConfig(random_seed=123).random_seed == 123


@pytest.mark.parametrize("random_seed", [True, 1.5, "42"])
def test_invalid_seed_rejected(random_seed):
    with pytest.raises(ValueError):
        ResponseOptimizationConfig(random_seed=random_seed)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_initial_assignment_probability_boundaries_allowed(probability):
    assert (
        ResponseOptimizationConfig(initial_assignment_probability=probability).initial_assignment_probability
        == probability
    )


@pytest.mark.parametrize("probability", [-0.01, 1.01, True])
def test_invalid_initial_assignment_probability_rejected(probability):
    with pytest.raises(ValueError):
        ResponseOptimizationConfig(initial_assignment_probability=probability)


def test_tournament_size_boundaries():
    assert ResponseOptimizationConfig(population_size=2, tournament_size=2).tournament_size == 2

    for value in (1, 3, True):
        with pytest.raises(ValueError):
            ResponseOptimizationConfig(population_size=2, tournament_size=value)


def test_elitism_count_boundaries():
    assert ResponseOptimizationConfig(population_size=2, elitism_count=0).elitism_count == 0
    assert ResponseOptimizationConfig(population_size=2, elitism_count=1).elitism_count == 1

    for value in (-1, 2, True):
        with pytest.raises(ValueError):
            ResponseOptimizationConfig(population_size=2, elitism_count=value)
