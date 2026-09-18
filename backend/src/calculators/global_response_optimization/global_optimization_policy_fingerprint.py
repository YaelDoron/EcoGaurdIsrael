"""Deterministic fingerprint of every policy/config that influences one
Global GA run's OUTCOME (Stage 6 of the Global Multi-Incident Optimizer
refactor, Task 17).

Covers GlobalResponseOptimizationConfig, SeverityDemandPolicy,
GlobalDemandScoringPolicy, GlobalAssignmentStabilityPolicy, and the
optimizer's own methodology/version - everything GlobalPlanningRefreshCoordinator
needs to decide "did the RULES change, not just the world" (a config/policy
version bump must trigger a re-optimization even when GlobalPlanningInput's
own content-fingerprint is unchanged). Pure - no DB, no randomness beyond
what the caller's config itself already carries (random_seed IS included,
deliberately: two different seeds are two different reproducible outcomes,
so a seed change must not be treated as a no-op).
"""
from __future__ import annotations

import hashlib
import json

from src.calculators.global_response_optimization.global_assignment_stability_policy import (
    GlobalAssignmentStabilityPolicy,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_response_optimization_config import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy


def compute_optimization_policy_fingerprint(
    config: GlobalResponseOptimizationConfig,
    demand_scoring_policy: GlobalDemandScoringPolicy,
    severity_demand_policy: SeverityDemandPolicy,
    stability_policy: GlobalAssignmentStabilityPolicy,
) -> str:
    payload = (
        ("optimizer_methodology", METHODOLOGY),
        ("optimizer_methodology_version", METHODOLOGY_VERSION),
        (
            "config",
            (
                config.population_size,
                config.generation_count,
                config.mutation_rate,
                config.crossover_rate,
                config.random_seed,
                config.eta_reference_seconds,
                config.initial_assignment_probability,
                config.tournament_size,
                config.elitism_count,
            ),
        ),
        (
            "demand_scoring_policy",
            (
                demand_scoring_policy.required_coverage_weight,
                demand_scoring_policy.desired_coverage_weight,
                demand_scoring_policy.predicted_risk_coverage_weight,
                demand_scoring_policy.severity_priority_weight,
                demand_scoring_policy.within_tier_component_cap,
                demand_scoring_policy.methodology,
                demand_scoring_policy.methodology_version,
            ),
        ),
        (
            "severity_demand_policy",
            (
                tuple(sorted((level.value, m, d) for level, (m, d) in (
                    (level, severity_demand_policy.demand_for_level(level))
                    for level in severity_demand_policy.minimum_by_level
                ))),
                severity_demand_policy.fallback_minimum_resources,
                severity_demand_policy.fallback_desired_resources,
                severity_demand_policy.methodology,
                severity_demand_policy.methodology_version,
            ),
        ),
        (
            "stability_policy",
            (
                stability_policy.stability_bonus_weight,
                stability_policy.methodology,
                stability_policy.methodology_version,
            ),
        ),
    )
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
