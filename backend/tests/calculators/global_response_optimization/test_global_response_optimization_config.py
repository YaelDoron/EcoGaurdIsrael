"""Tests for GlobalResponseOptimizationConfig (Stage 4, Task 9)."""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_response_optimization_config import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    GlobalResponseOptimizationConfig,
)


def test_default_config_is_valid():
    config = GlobalResponseOptimizationConfig()
    assert config.population_size >= 2
    assert config.generation_count >= 1


def test_methodology_is_distinct_from_legacy():
    assert METHODOLOGY == "global_genetic_resource_allocation"
    assert METHODOLOGY != "GENETIC_RESOURCE_ALLOCATION"
    assert METHODOLOGY_VERSION == "1.0"


@pytest.mark.parametrize(
    "overrides",
    [
        {"population_size": 1},
        {"generation_count": 0},
        {"mutation_rate": 1.5},
        {"crossover_rate": -0.1},
        {"eta_reference_seconds": 0},
        {"initial_assignment_probability": 2.0},
        {"tournament_size": 1},
        {"elitism_count": -1},
    ],
)
def test_rejects_invalid_fields(overrides):
    with pytest.raises(ValueError):
        GlobalResponseOptimizationConfig(**overrides)


def test_is_frozen_and_reproducible_as_a_value_object():
    a = GlobalResponseOptimizationConfig(random_seed=5)
    b = GlobalResponseOptimizationConfig(random_seed=5)
    assert a == b
