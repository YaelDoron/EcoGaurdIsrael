"""Demand-aware global fitness scoring (Stage 5, Tasks 14-16, 35; soft
assignment stability added Stage 6, Tasks 12-13).

For a covered slot:

    value = tier_weight(slot) + severity_component(slot) + eta_priority_component(slot) + stability_component(slot, resource_id)

tier_weight puts the slot in one of three classes (REQUIRED suppression >
DESIRED suppression > PREDICTED_RISK optional coverage - see
GlobalDemandScoringPolicy for exactly why the ordering can never invert).
severity_component breaks ties between competing incidents under REQUIRED-
slot scarcity using SeverityDemandPolicy's configured severity_rank.
eta_priority_component is `target.priority_score * eta_factor(eta)` - the
EXACT Stage 4/legacy formula, reused unchanged (Task 35: this is the one
and only place `target.priority_score` - which already embeds
ACTIVE_FIRE_BASE_PRIORITY + severity_score, see response_target_calculator.py -
enters scoring; it is never added a second time), clamped so it can only
ever differentiate WITHIN a tier.

stability_component (Stage 6) adds `GlobalAssignmentStabilityPolicy.
stability_bonus_weight` when the resource being scored currently holds a
PLANNED (still-reassignable) commitment to the SAME target this slot
belongs to - "keep doing what you were already doing" is worth this many
points, so the GA only moves a tentatively-committed resource elsewhere
when the alternative's genuine improvement exceeds the bonus. It never
applies to a DISPATCHED (hard-locked) resource - that resource cannot be
assigned anywhere else in the first place (see global_hard_dispatch_lock.py),
so no scoring pressure is needed to keep it in place.

Both eta_priority_component and stability_component are folded into
__init__'s cross-tier safety-margin validation (their combined maximum is
what GlobalDemandScoringPolicy's own within_tier_component_cap must exceed
by the required margin) - so a soft stability preference can never let a
lower tier (e.g. optional PREDICTED_RISK coverage) outscore a higher one
(e.g. feasible REQUIRED suppression), exactly like Stage 5's own guarantee.

fitness_score = sum of value(...) over every covered slot (Task 14: one
flat global sum, never computed per event and combined). This is a
deliberately UNBOUNDED tiered ranking value for the GA's own selection
pressure, not a percentage - coverage_score (a true 0-100 metric) is
reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.calculators.global_response_optimization.global_assignment_stability_policy import (
    GlobalAssignmentStabilityPolicy,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    GlobalResponseOptimizationProblem,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.models.dispatch_state import DispatchState
from src.models.global_allocation_slot import GlobalAllocationSlot
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.response_target_type import ResponseTargetType


class GlobalResponsePlanScoringError(ValueError):
    """Raised when a chromosome cannot be scored against its problem."""


@dataclass(frozen=True)
class GlobalScoreBreakdown:
    """Fitness/coverage breakdown for one scored chromosome."""

    fitness_score: float
    coverage_score: float
    average_eta_seconds: float | None
    covered_slot_count: int
    total_slot_count: int
    covered_slot_ids: tuple[str, ...]
    uncovered_slot_ids: tuple[str, ...]


class GlobalResponsePlanScorer:
    """Evaluate one resource-indexed chromosome's demand-aware global fitness."""

    def __init__(
        self,
        config: GlobalResponseOptimizationConfig | None = None,
        demand_scoring_policy: GlobalDemandScoringPolicy | None = None,
        severity_demand_policy: SeverityDemandPolicy | None = None,
        stability_policy: GlobalAssignmentStabilityPolicy | None = None,
    ) -> None:
        self._config = config or GlobalResponseOptimizationConfig()
        self._demand_scoring_policy = demand_scoring_policy or GlobalDemandScoringPolicy()
        self._severity_demand_policy = severity_demand_policy or SeverityDemandPolicy()
        self._stability_policy = stability_policy or GlobalAssignmentStabilityPolicy()
        self._validate_combined_within_tier_margin()

    def _validate_combined_within_tier_margin(self) -> None:
        """Stage 6: re-derive GlobalDemandScoringPolicy's own cross-tier
        safety proof with stability_bonus_weight folded into the maximum
        within-tier total, so adding a soft stability preference can never
        by itself let a lower tier outscore a higher one."""
        policy = self._demand_scoring_policy
        max_within_tier = (
            policy.severity_priority_weight * policy._MAX_SEVERITY_RANK
            + policy.within_tier_component_cap
            + self._stability_policy.stability_bonus_weight
        )
        if policy.required_coverage_weight - policy.desired_coverage_weight <= max_within_tier:
            raise ValueError(
                "required_coverage_weight must exceed desired_coverage_weight by more than severity_priority_weight*3 "
                f"+ within_tier_component_cap + stability_bonus_weight ({max_within_tier!r}), so a REQUIRED slot can "
                "never score below a DESIRED slot even with a soft stability bonus applied."
            )
        if policy.desired_coverage_weight - policy.predicted_risk_coverage_weight <= max_within_tier:
            raise ValueError(
                "desired_coverage_weight must exceed predicted_risk_coverage_weight by more than "
                f"severity_priority_weight*3 + within_tier_component_cap + stability_bonus_weight ({max_within_tier!r}), "
                "so a DESIRED slot can never score below a PREDICTED_RISK slot even with a soft stability bonus applied."
            )

    def eta_factor(self, eta_seconds: float) -> float:
        if isinstance(eta_seconds, bool) or not isinstance(eta_seconds, Real) or not math.isfinite(eta_seconds) or eta_seconds < 0:
            raise GlobalResponsePlanScoringError(
                f"eta_seconds must be a finite non-negative number, got {eta_seconds!r}"
            )
        return 1.0 / (1.0 + float(eta_seconds) / self._config.eta_reference_seconds)

    def slot_value(
        self,
        problem: GlobalResponseOptimizationProblem,
        slot: GlobalAllocationSlot,
        eta_seconds: float,
        resource_id: str | None = None,
    ) -> float:
        """The value of covering `slot` at the given ETA - see module docstring for the exact formula."""
        target = problem.targets_by_id[slot.response_target_id]
        policy = self._demand_scoring_policy

        if target.target_type is ResponseTargetType.PREDICTED_RISK:
            tier_weight = policy.predicted_risk_coverage_weight
        elif slot.required:
            tier_weight = policy.required_coverage_weight
        else:
            tier_weight = policy.desired_coverage_weight

        demand = problem.incident_demands_by_event.get(slot.fire_event_id)
        severity_level = demand.severity_level if demand is not None else None
        # No known severity (fallback demand): treated as the lowest severity
        # rank - it is still a REQUIRED slot (fallback minimum >= 1 keeps it
        # covered before any DESIRED slot), but never assumed to outrank a
        # confirmed LOW/MODERATE/HIGH/CRITICAL incident under scarcity.
        severity_rank = self._severity_demand_policy.severity_rank(severity_level) if severity_level is not None else 0
        severity_component = policy.severity_priority_weight * severity_rank

        raw_eta_priority = target.priority_score * self.eta_factor(eta_seconds)
        eta_priority_component = policy.clamp_eta_priority_component(raw_eta_priority)

        stability_component = self._stability_component(problem, slot, resource_id)

        return tier_weight + severity_component + eta_priority_component + stability_component

    def _stability_component(
        self, problem: GlobalResponseOptimizationProblem, slot: GlobalAllocationSlot, resource_id: str | None
    ) -> float:
        if resource_id is None:
            return 0.0
        assignment = problem.current_assignment_by_resource.get(resource_id)
        if assignment is None or assignment.dispatch_state is not DispatchState.PLANNED:
            return 0.0
        if assignment.response_target_id != slot.response_target_id:
            return 0.0
        return self._stability_policy.stability_bonus_weight

    def evaluate(
        self,
        problem: GlobalResponseOptimizationProblem,
        chromosome: GlobalResponsePlanChromosome,
    ) -> GlobalScoreBreakdown:
        if not isinstance(problem, GlobalResponseOptimizationProblem):
            raise GlobalResponsePlanScoringError(
                f"problem must be a GlobalResponseOptimizationProblem, got {problem!r}"
            )
        if not isinstance(chromosome, GlobalResponsePlanChromosome):
            raise GlobalResponsePlanScoringError(
                f"chromosome must be a GlobalResponsePlanChromosome, got {chromosome!r}"
            )
        if chromosome.resource_ids != problem.resource_ids:
            raise GlobalResponsePlanScoringError(
                "chromosome resource_ids must exactly match the problem's resource_ids."
            )

        covered_slot_ids: set[str] = set()
        fitness_score = 0.0
        eta_values: list[float] = []

        for resource_id, slot_id in zip(chromosome.resource_ids, chromosome.genes):
            if slot_id is None:
                continue
            if slot_id in covered_slot_ids:
                raise GlobalResponsePlanScoringError(f"slot {slot_id!r} is assigned more than once.")
            slot = problem.slots_by_id.get(slot_id)
            if slot is None:
                raise GlobalResponsePlanScoringError(f"chromosome references unknown slot_id {slot_id!r}.")
            route = problem.route_matrix.get(resource_id, slot.response_target_id)
            if route is None:
                raise GlobalResponsePlanScoringError(
                    f"no feasible route exists for resource {resource_id!r} -> slot {slot_id!r}."
                )

            fitness_score += self.slot_value(problem, slot, route.eta_seconds, resource_id)
            eta_values.append(route.eta_seconds)
            covered_slot_ids.add(slot_id)

        total_slot_count = len(problem.slots)
        coverage_score = 100.0 * len(covered_slot_ids) / total_slot_count if total_slot_count > 0 else 0.0
        average_eta_seconds = sum(eta_values) / len(eta_values) if eta_values else None
        uncovered_slot_ids = tuple(
            slot.slot_id for slot in problem.slots if slot.slot_id not in covered_slot_ids
        )

        return GlobalScoreBreakdown(
            fitness_score=fitness_score,
            coverage_score=coverage_score,
            average_eta_seconds=average_eta_seconds,
            covered_slot_count=len(covered_slot_ids),
            total_slot_count=total_slot_count,
            covered_slot_ids=tuple(sorted(covered_slot_ids)),
            uncovered_slot_ids=uncovered_slot_ids,
        )
