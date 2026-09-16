"""Pure deterministic optimized-vs-baseline plan comparison.

Implements User Story 5.3 Task 3: compares an already-evaluated optimized
(GA) plan against Task 2's already-evaluated `BaselinePlanResult`. This
module computes ONLY the two genuinely comparison-specific metrics
(`score_difference`, `improvement_percentage`); every other value on
`PlanComparison` is copied unchanged from data the optimized plan and the
baseline evaluation already produced.

This module MUST NOT and does NOT:
- run Dijkstra, the Genetic Algorithm, or the greedy baseline again
  (Task 1's `BaselinePlanCalculator` is not imported here);
- call the shared scorer again (Task 2's `BaselinePlanEvaluator` is not
  imported here -- both plans must already be evaluated before calling
  this calculator);
- recompute coverage, average ETA, target priority, severity, spread, or
  routes/ETA -- those numbers are read directly off the already-evaluated
  optimized and baseline score data.

As of Task 3, Company 2's real `ResponsePlan`/`StoredResponsePlan`/
`PlanScoreBreakdown`/`ResponsePlanScorer` still do not exist in this
codebase (same finding as Task 1/2 -- see `baseline_plan_calculator.py`
and `baseline_plan_evaluator.py`). Rather than fabricate a competing
`ResponsePlan` implementation, this module defines the minimal
Company-3-owned pure input `OptimizedPlanEvaluation`, exposing only the
fields this comparison actually needs (identifiers + score, reusing
Task 2's `PlanScoreBreakdownLike` structural shape for the score). Once
Company 2's real `ResponsePlan` is merged, adapting requires only
constructing an `OptimizedPlanEvaluation` from its fields at the call
site -- no change to `BaselinePlanComparisonCalculator`.

`PlanComparison`, by contrast, belongs to Company 3 under the frozen
Task 0 contract, so it is implemented here directly with its exact frozen
field set.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.calculators.baseline_plan.baseline_plan_evaluator import (
    BaselinePlanResult,
    PlanScoreBreakdownLike,
)


@dataclass(frozen=True)
class OptimizedPlanEvaluation:
    """Minimal Company-3-owned pure input representing the already-evaluated optimized plan.

    NOT a competing implementation of the frozen `ResponsePlan`/
    `StoredResponsePlan` (Company 2, US 5.2) -- those do not exist in this
    branch yet. Exposes only the identifiers and already-computed score
    this comparison needs; nothing here is recalculated by Task 3.
    """

    optimized_plan_id: int
    fire_event_id: int
    route_planning_run_id: int
    response_target_set_id: int
    score: PlanScoreBreakdownLike

    def __post_init__(self) -> None:
        _validate_positive_int("optimized_plan_id", self.optimized_plan_id)
        _validate_positive_int("fire_event_id", self.fire_event_id)
        _validate_positive_int("route_planning_run_id", self.route_planning_run_id)
        _validate_positive_int("response_target_set_id", self.response_target_set_id)


@dataclass(frozen=True)
class PlanComparison:
    """Company 3's implementation of the frozen `PlanComparison` contract (Task 0)."""

    fire_event_id: int
    optimized_plan_id: int
    route_planning_run_id: int
    response_target_set_id: int
    optimized_score: float
    baseline_score: float
    optimized_coverage_score: float
    baseline_coverage_score: float
    optimized_average_eta_seconds: float | None
    baseline_average_eta_seconds: float | None
    score_difference: float
    improvement_percentage: float | None

    def __post_init__(self) -> None:
        _validate_positive_int("fire_event_id", self.fire_event_id)
        _validate_positive_int("optimized_plan_id", self.optimized_plan_id)
        _validate_positive_int("route_planning_run_id", self.route_planning_run_id)
        _validate_positive_int("response_target_set_id", self.response_target_set_id)


class BaselinePlanComparisonCalculator:
    """Pure comparison of an already-evaluated optimized plan against a `BaselinePlanResult`."""

    def compare(
        self,
        *,
        optimized: OptimizedPlanEvaluation,
        baseline: BaselinePlanResult,
    ) -> PlanComparison:
        """Compare two already-evaluated plans from the SAME planning snapshot.

        Requires `optimized.route_planning_run_id == baseline.route_planning_run_id`
        and `optimized.response_target_set_id == baseline.response_target_set_id`
        -- comparing plans built from different snapshots (different
        `ResponseTargetSet`/resource snapshot/routes/ETA matrix) would not be a
        fair apples-to-apples comparison, so it is rejected rather than silently
        allowed or resolved.

        Every metric other than `score_difference`/`improvement_percentage` is
        copied unchanged from `optimized.score`/`baseline.score`; none of
        coverage, average ETA, or the total score is recalculated here.
        """
        if not isinstance(optimized, OptimizedPlanEvaluation):
            raise ValueError(f"optimized must be an OptimizedPlanEvaluation, got {optimized!r}")
        if not isinstance(baseline, BaselinePlanResult):
            raise ValueError(f"baseline must be a BaselinePlanResult, got {baseline!r}")

        if optimized.route_planning_run_id != baseline.route_planning_run_id:
            raise ValueError(
                "optimized and baseline must share the same route_planning_run_id for a "
                f"valid comparison, got optimized={optimized.route_planning_run_id!r} "
                f"baseline={baseline.route_planning_run_id!r}"
            )
        if optimized.response_target_set_id != baseline.response_target_set_id:
            raise ValueError(
                "optimized and baseline must share the same response_target_set_id for a "
                f"valid comparison, got optimized={optimized.response_target_set_id!r} "
                f"baseline={baseline.response_target_set_id!r}"
            )

        optimized_score = optimized.score.total_score
        baseline_score = baseline.score.total_score

        return PlanComparison(
            fire_event_id=optimized.fire_event_id,
            optimized_plan_id=optimized.optimized_plan_id,
            route_planning_run_id=optimized.route_planning_run_id,
            response_target_set_id=optimized.response_target_set_id,
            optimized_score=optimized_score,
            baseline_score=baseline_score,
            optimized_coverage_score=optimized.score.coverage_score,
            baseline_coverage_score=baseline.score.coverage_score,
            optimized_average_eta_seconds=optimized.score.average_eta_seconds,
            baseline_average_eta_seconds=baseline.score.average_eta_seconds,
            score_difference=optimized_score - baseline_score,
            improvement_percentage=_improvement_percentage(optimized_score, baseline_score),
        )


def _improvement_percentage(optimized_score: float, baseline_score: float) -> float | None:
    if baseline_score == 0:
        return None
    return ((optimized_score - baseline_score) / baseline_score) * 100


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
