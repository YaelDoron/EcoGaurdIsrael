"""Tests for compute_optimization_policy_fingerprint (Stage 6, Task 17)."""
from __future__ import annotations

from src.calculators.global_response_optimization.global_assignment_stability_policy import (
    GlobalAssignmentStabilityPolicy,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_optimization_policy_fingerprint import (
    compute_optimization_policy_fingerprint,
)
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy


def _fingerprint(**overrides):
    defaults = dict(
        config=GlobalResponseOptimizationConfig(),
        demand_scoring_policy=GlobalDemandScoringPolicy(),
        severity_demand_policy=SeverityDemandPolicy(),
        stability_policy=GlobalAssignmentStabilityPolicy(),
    )
    defaults.update(overrides)
    return compute_optimization_policy_fingerprint(**defaults)


def test_same_policies_produce_the_same_fingerprint():
    assert _fingerprint() == _fingerprint()


def test_is_a_64_char_hex_sha256_digest():
    fingerprint = _fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)  # raises ValueError if not valid hex


def test_config_random_seed_change_changes_the_fingerprint():
    a = _fingerprint(config=GlobalResponseOptimizationConfig(random_seed=1))
    b = _fingerprint(config=GlobalResponseOptimizationConfig(random_seed=2))
    assert a != b


def test_config_population_size_change_changes_the_fingerprint():
    a = _fingerprint(config=GlobalResponseOptimizationConfig(population_size=10))
    b = _fingerprint(config=GlobalResponseOptimizationConfig(population_size=20))
    assert a != b


def test_demand_scoring_policy_change_changes_the_fingerprint():
    a = _fingerprint(demand_scoring_policy=GlobalDemandScoringPolicy())
    b = _fingerprint(demand_scoring_policy=GlobalDemandScoringPolicy(severity_priority_weight=20.0))
    assert a != b


def test_stability_policy_change_changes_the_fingerprint():
    a = _fingerprint(stability_policy=GlobalAssignmentStabilityPolicy())
    b = _fingerprint(stability_policy=GlobalAssignmentStabilityPolicy(stability_bonus_weight=99.0))
    assert a != b


def test_severity_demand_policy_change_changes_the_fingerprint():
    from src.models.fire_severity_level import FireSeverityLevel

    a = _fingerprint(severity_demand_policy=SeverityDemandPolicy())
    b = _fingerprint(
        severity_demand_policy=SeverityDemandPolicy(
            minimum_by_level={
                FireSeverityLevel.LOW: 1, FireSeverityLevel.MODERATE: 1,
                FireSeverityLevel.HIGH: 2, FireSeverityLevel.CRITICAL: 4,
            },
        )
    )
    assert a != b
