"""Connects Task 1's deterministic greedy baseline to shared Epic 5 scoring.

Implements User Story 5.3 Task 2: shared scoring integration. This module
never computes a score itself. Company 2 owns the fitness/scoring
mathematics (`ResponsePlanScorer.evaluate(...) -> PlanScoreBreakdown`, the
same evaluator the GA must use), and this evaluator only:

1. runs Task 1's frozen `BaselinePlanCalculator.allocate(...)` to get the
   deterministic baseline allocation;
2. repackages that allocation plus the inputs it was built from into a
   scoring context;
3. calls an externally supplied scorer exactly once with that context;
4. returns the scorer's result unchanged, alongside the identifiers and
   assignments the caller supplied/Task 1 produced.

As of Task 2, none of `ResponsePlanScorer`, `PlanScoreBreakdown`, or
`ResponseAction` exist yet in this codebase (US 5.1/5.2 not merged -- see
`baseline_plan_calculator.py` for the same finding re: routing contracts).
The frozen Task 0 contract guarantees `ResponsePlanScorer.evaluate(...) ->
PlanScoreBreakdown` but does not freeze `evaluate`'s argument list, so this
module does not guess it. Instead it defines a minimal structural boundary:

- `BaselinePlanScorer` (a `Protocol`) is the scoring port. It takes exactly
  one argument, `BaselinePlanScoringContext`, built entirely from data
  Task 1 already produced or received -- nothing is recalculated
  (no ETA, distance, reachability, priority, severity, or spread).
- `PlanScoreBreakdownLike` (a `Protocol`) types "whatever the injected
  scorer returns" using the frozen `PlanScoreBreakdown` field names,
  without importing or redefining Company 2's concrete dataclass.
- `BaselinePlanResult` is Company 3's implementation of the frozen
  `BaselinePlanResult` contract. Its `actions` field uses Task 1's
  `BaselineAssignment` (already structurally identical to the frozen
  `ResponseAction`: `resource_id`, `response_target_id`,
  `route_result_id`) and its `score` field is typed via
  `PlanScoreBreakdownLike` rather than a competing concrete class.

Once Company 2's real `ResponsePlanScorer`/`PlanScoreBreakdown` and Company
1's `ResponseAction` are merged, wiring the real scorer requires only an
adapter that implements `BaselinePlanScorer` -- no change to
`BaselinePlanEvaluator` or `baseline_plan_calculator.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.calculators.baseline_plan.baseline_plan_calculator import (
    BaselineAssignment,
    BaselinePlanAllocation,
    BaselinePlanCalculator,
    RouteCandidate,
    TargetOrder,
)
from src.calculators.baseline_plan.baseline_plan_config import (
    BASELINE_PLAN_METHODOLOGY,
    BASELINE_PLAN_METHODOLOGY_VERSION,
)


class PlanScoreBreakdownLike(Protocol):
    """Structural shape of the frozen `PlanScoreBreakdown` contract (Task 0 / Company 2).

    Not a competing implementation: this exists only so `BaselinePlanResult`
    can be type-annotated without importing Company 2's concrete dataclass,
    which does not exist in this branch yet. Task 2 never reads or
    recomputes these fields; it passes the scorer's return value through
    unchanged.
    """

    total_score: float
    coverage_score: float
    average_eta_seconds: float | None
    covered_target_count: int
    total_target_count: int
    uncovered_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class BaselinePlanScoringContext:
    """Everything the shared scorer needs, repackaged from data Task 1 already has.

    Every field here is either a caller-supplied identifier or something
    `BaselinePlanCalculator.allocate(...)` already produced/consumed --
    nothing is recalculated.
    """

    route_planning_run_id: int
    response_target_set_id: int
    targets: tuple[TargetOrder, ...]
    route_candidates: tuple[RouteCandidate, ...]
    allocation: BaselinePlanAllocation


class BaselinePlanScorer(Protocol):
    """Structural port for the shared `ResponsePlanScorer.evaluate(...)`.

    The exact argument list of Company 2's real `evaluate` is not frozen by
    Task 0, so this port fixes only ONE stable seam: a single
    `BaselinePlanScoringContext` argument. Adapting to the real scorer's
    eventual signature is an adapter-level concern, not a change to this
    evaluator.
    """

    def evaluate(self, context: BaselinePlanScoringContext) -> PlanScoreBreakdownLike: ...


@dataclass(frozen=True)
class BaselinePlanResult:
    """Company 3's implementation of the frozen `BaselinePlanResult` contract.

    `actions` uses Task 1's `BaselineAssignment` (structurally identical to
    the frozen `ResponseAction`) and `score` is whatever the injected
    scorer returned, typed structurally via `PlanScoreBreakdownLike`.
    """

    route_planning_run_id: int
    response_target_set_id: int
    actions: tuple[BaselineAssignment, ...]
    score: PlanScoreBreakdownLike
    methodology: str = BASELINE_PLAN_METHODOLOGY
    methodology_version: str = BASELINE_PLAN_METHODOLOGY_VERSION

    def __post_init__(self) -> None:
        _validate_positive_int("route_planning_run_id", self.route_planning_run_id)
        _validate_positive_int("response_target_set_id", self.response_target_set_id)
        if self.methodology != BASELINE_PLAN_METHODOLOGY:
            raise ValueError(
                f"methodology must be {BASELINE_PLAN_METHODOLOGY!r}, got {self.methodology!r}"
            )
        if self.methodology_version != BASELINE_PLAN_METHODOLOGY_VERSION:
            raise ValueError(
                f"methodology_version must be {BASELINE_PLAN_METHODOLOGY_VERSION!r}, "
                f"got {self.methodology_version!r}"
            )


class BaselinePlanEvaluator:
    """Runs Task 1's greedy baseline, then scores it via an injected shared scorer."""

    def __init__(self, *, calculator: BaselinePlanCalculator | None = None) -> None:
        self._calculator = calculator if calculator is not None else BaselinePlanCalculator()

    def evaluate(
        self,
        *,
        route_planning_run_id: int,
        response_target_set_id: int,
        targets: Sequence[TargetOrder],
        route_candidates: Sequence[RouteCandidate],
        scorer: BaselinePlanScorer,
    ) -> BaselinePlanResult:
        """Allocate the deterministic baseline, then score it exactly once.

        Never computes a score locally: `scorer.evaluate(...)` is called
        exactly once and its return value is preserved unchanged.
        """
        _validate_positive_int("route_planning_run_id", route_planning_run_id)
        _validate_positive_int("response_target_set_id", response_target_set_id)

        allocation = self._calculator.allocate(targets=targets, route_candidates=route_candidates)

        context = BaselinePlanScoringContext(
            route_planning_run_id=route_planning_run_id,
            response_target_set_id=response_target_set_id,
            targets=tuple(targets),
            route_candidates=tuple(route_candidates),
            allocation=allocation,
        )
        score = scorer.evaluate(context)

        return BaselinePlanResult(
            route_planning_run_id=route_planning_run_id,
            response_target_set_id=response_target_set_id,
            actions=allocation.assignments,
            score=score,
        )


def _validate_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
