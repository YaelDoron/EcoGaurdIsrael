"""EcoGuard demand-aware global scoring policy (Stage 5 of the Global
Multi-Incident Optimizer refactor, Tasks 14-16).

The scorer values a covered slot by three additive, explicitly separated
components, so the priority hierarchy stays easy to explain rather than
buried in one opaque number:

    value = tier_weight(slot) + severity_component(slot) + eta_priority_component(slot)

`tier_weight` puts a covered slot into one of three priority CLASSES:
    1. REQUIRED suppression coverage  (highest)
    2. DESIRED  suppression coverage
    3. PREDICTED_RISK optional coverage (lowest)
`severity_component` breaks ties BETWEEN incidents competing for the same
class under scarcity (Task 15) using SeverityDemandPolicy's configured
severity_rank (0=LOW..3=CRITICAL) - a decision-support ordering, not a
claim about real dispatch doctrine.
`eta_priority_component` is `target.priority_score * eta_factor(eta)`
(the exact Stage 4/legacy formula, reused - not recomputed or
double-applied), CLAMPED to `within_tier_component_cap` so it can only ever
differentiate WITHIN a class, never cross into a different one, regardless
of how large an upstream target.priority_score happens to be.

Why the ordering can never invert (validated in __post_init__, not merely
asserted): the maximum possible severity_component
(`severity_priority_weight * 3`) plus the capped eta/priority component
(`within_tier_component_cap`) is REQUIRED to be strictly smaller than the
gap between adjacent tier weights. So even the worst-scoring REQUIRED slot
outscores the best-scoring DESIRED slot, and the worst-scoring DESIRED slot
outscores the best-scoring PREDICTED_RISK slot, unconditionally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

METHODOLOGY = "ecoguard_demo_demand_scoring_policy"
METHODOLOGY_VERSION = "1.0"

DEFAULT_REQUIRED_COVERAGE_WEIGHT = 1_000_000.0
DEFAULT_DESIRED_COVERAGE_WEIGHT = 10_000.0
DEFAULT_PREDICTED_RISK_COVERAGE_WEIGHT = 100.0
DEFAULT_SEVERITY_PRIORITY_WEIGHT = 10.0
DEFAULT_WITHIN_TIER_COMPONENT_CAP = 200.0


@dataclass(frozen=True)
class GlobalDemandScoringPolicy:
    """Versioned, validated tier-weight configuration for Stage 5's global scorer."""

    required_coverage_weight: float = DEFAULT_REQUIRED_COVERAGE_WEIGHT
    desired_coverage_weight: float = DEFAULT_DESIRED_COVERAGE_WEIGHT
    predicted_risk_coverage_weight: float = DEFAULT_PREDICTED_RISK_COVERAGE_WEIGHT
    severity_priority_weight: float = DEFAULT_SEVERITY_PRIORITY_WEIGHT
    within_tier_component_cap: float = DEFAULT_WITHIN_TIER_COMPONENT_CAP
    methodology: str = METHODOLOGY
    methodology_version: str = METHODOLOGY_VERSION

    # Severity rank spans 0..3 (LOW..CRITICAL) - see SeverityDemandPolicy.severity_rank.
    _MAX_SEVERITY_RANK = 3

    def __post_init__(self) -> None:
        for field_name, value in (
            ("required_coverage_weight", self.required_coverage_weight),
            ("desired_coverage_weight", self.desired_coverage_weight),
            ("predicted_risk_coverage_weight", self.predicted_risk_coverage_weight),
            ("severity_priority_weight", self.severity_priority_weight),
            ("within_tier_component_cap", self.within_tier_component_cap),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a positive finite number, got {value!r}")

        max_within_tier = self.severity_priority_weight * self._MAX_SEVERITY_RANK + self.within_tier_component_cap
        if self.required_coverage_weight - self.desired_coverage_weight <= max_within_tier:
            raise ValueError(
                "required_coverage_weight must exceed desired_coverage_weight by more than "
                f"severity_priority_weight*3 + within_tier_component_cap ({max_within_tier!r}), so a REQUIRED "
                "slot can never score below a DESIRED slot."
            )
        if self.desired_coverage_weight - self.predicted_risk_coverage_weight <= max_within_tier:
            raise ValueError(
                "desired_coverage_weight must exceed predicted_risk_coverage_weight by more than "
                f"severity_priority_weight*3 + within_tier_component_cap ({max_within_tier!r}), so a DESIRED "
                "slot can never score below a PREDICTED_RISK slot."
            )

        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(f"methodology_version must be a non-empty string, got {self.methodology_version!r}")

    def clamp_eta_priority_component(self, raw_value: float) -> float:
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real) or not math.isfinite(raw_value) or raw_value < 0:
            raise ValueError(f"raw_value must be a non-negative finite number, got {raw_value!r}")
        return min(float(raw_value), self.within_tier_component_cap)
