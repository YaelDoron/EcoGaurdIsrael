"""EcoGuard soft assignment-stability policy (Stage 6 of the Global
Multi-Incident Optimizer refactor, Tasks 12-13).

Distinct from the HARD dispatch lock (see global_response_optimization_problem.py
and dispatch_state.py): this policy only ever influences a resource whose
CURRENT commitment is still PLANNED (tentative/reassignable). It never
applies to a DISPATCHED (hard-locked) resource - that resource's feasible
allele set is already structurally restricted so it cannot leave its
FireEvent at all, independent of any scoring weight.

`stability_bonus_weight` is added to a slot's value (see
GlobalResponsePlanScorer.slot_value) when the resource evaluated for that
slot currently holds a PLANNED commitment to the SAME FireEvent's ACTIVE_FIRE
target - i.e. "keep doing what you were already doing" is worth this many
points, so the GA only moves a tentatively-committed resource elsewhere when
the alternative's genuine improvement (severity/eta) exceeds this bonus.

Like GlobalDemandScoringPolicy (Stage 5), this is one more additive,
strictly WITHIN-TIER component - never large enough to let a lower tier
(e.g. PREDICTED_RISK) outscore a higher one (e.g. REQUIRED suppression).
__post_init__ folds `stability_bonus_weight` into the SAME cross-tier safety
margin GlobalDemandScoringPolicy proves for itself, so this can only ever be
validated together with a demand_scoring_policy instance (see
GlobalResponsePlanScorer, which owns both policies and re-validates their
combined margin at construction time) - this class alone only validates its
own weight is a sane, positive, finite number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

METHODOLOGY = "ecoguard_demo_assignment_stability_policy"
METHODOLOGY_VERSION = "1.0"

# Sized to damp micro-flickering between near-identical candidates, never to
# entrench a materially worse (i.e. meaningfully farther) one. Against the
# quadratic eta_factor (see GlobalResponsePlanScorer.eta_factor,
# eta_reference_seconds=900s default), even a conservative multi-minute ETA
# advantage clears this easily: e.g. a target already at the max active-fire
# priority (200.0) reachable at 20 vs. 27 minutes (a 7-minute gap late in the
# curve, where the quadratic penalty's marginal bite is at its weakest) still
# swings eta_priority_component by ~25 points - several times this weight.
# At the low end (a low-priority PREDICTED_RISK target, priority_score close
# to its ~51 floor) the same 7-minute gap still swings the component by
# several points, comfortably above this weight. A gap of ~15 seconds -
# genuine routing noise rather than a real distance difference - swings the
# component by well under this weight even at max priority; only once a gap
# approaches a real minute or more does it start to matter, which is exactly
# the "keep doing what you were already doing" case this bonus exists for.
DEFAULT_STABILITY_BONUS_WEIGHT = 5.0


@dataclass(frozen=True)
class GlobalAssignmentStabilityPolicy:
    """Versioned, validated soft-stability configuration for the Stage 6 global scorer."""

    stability_bonus_weight: float = DEFAULT_STABILITY_BONUS_WEIGHT
    methodology: str = METHODOLOGY
    methodology_version: str = METHODOLOGY_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.stability_bonus_weight, bool)
            or not isinstance(self.stability_bonus_weight, Real)
            or not math.isfinite(self.stability_bonus_weight)
            or self.stability_bonus_weight <= 0
        ):
            raise ValueError(
                f"stability_bonus_weight must be a positive finite number, got {self.stability_bonus_weight!r}"
            )
        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be a non-empty string, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(f"methodology_version must be a non-empty string, got {self.methodology_version!r}")
