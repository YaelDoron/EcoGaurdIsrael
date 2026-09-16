"""Real production adapter satisfying BaselinePlanScorer via the shared ResponsePlanScorer.

TRANSLATION ONLY. `BaselineComparisonService` already loaded and validated the
exact ResponsePlan/RoutePlanningRun/ResponseTargetSet snapshot before calling
`BaselinePlanEvaluator.evaluate(...)`, which carries fire_event_id, each
target's target_type/priority_score, and each route's distance_meters through
into `BaselinePlanScoringContext` (see the Task 6.1 notes in
baseline_plan_calculator.py / baseline_plan_evaluator.py) without any fresh
repository lookup. This adapter only reshapes that already-complete context
into Company 2's existing `ResponseOptimizationInput`/`ResponseAction` shape
and delegates to the real `ResponsePlanScorer.evaluate(...)` exactly once. No
scoring formula, coverage rule, ETA penalty, or weighting constant is
duplicated here.

`resources` is derived from the resource_ids already present on
`context.route_candidates` rather than adding a separate resource_ids field
to `BaselinePlanScoringContext`: `ResponsePlanScorer.evaluate()` only uses
`resources` to confirm each action's resource_id is a known one, and every
assignment in `context.allocation` was chosen by Task 1's calculator from
`route_candidates` in the first place, so this set is always sufficient.
"""
from __future__ import annotations

from src.calculators.baseline_plan.baseline_plan_evaluator import BaselinePlanScoringContext
from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer
from src.models.optimization_resource import OptimizationResource
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.plan_score_breakdown import PlanScoreBreakdown
from src.models.response_action import ResponseAction
from src.models.response_optimization_input import ResponseOptimizationInput


class ResponsePlanBaselineScorerAdapter:
    """Bridges US 5.3's BaselinePlanScorer port to the real, shared ResponsePlanScorer."""

    def __init__(self, scorer: ResponsePlanScorer) -> None:
        self._scorer = scorer

    def evaluate(self, context: BaselinePlanScoringContext) -> PlanScoreBreakdown:
        """Score one baseline allocation, delegating to ResponsePlanScorer exactly once."""
        if not isinstance(context, BaselinePlanScoringContext):
            raise ValueError(f"context must be a BaselinePlanScoringContext, got {context!r}")

        optimization_input = ResponseOptimizationInput(
            fire_event_id=context.fire_event_id,
            response_target_set_id=context.response_target_set_id,
            route_planning_run_id=context.route_planning_run_id,
            targets=self._to_optimization_targets(context),
            resources=self._to_optimization_resources(context),
            route_options=self._to_route_options(context),
        )
        actions = tuple(
            ResponseAction(
                resource_id=assignment.resource_id,
                response_target_id=assignment.response_target_id,
                route_result_id=assignment.route_result_id,
            )
            for assignment in context.allocation.assignments
        )
        return self._scorer.evaluate(optimization_input, actions)

    @staticmethod
    def _to_optimization_targets(context: BaselinePlanScoringContext) -> tuple[OptimizationTarget, ...]:
        return tuple(
            OptimizationTarget(
                response_target_id=target.response_target_id,
                target_order=target.target_order,
                target_type=target.target_type,
                priority_score=target.priority_score,
            )
            for target in context.targets
        )

    @staticmethod
    def _to_optimization_resources(context: BaselinePlanScoringContext) -> tuple[OptimizationResource, ...]:
        resource_ids = {candidate.resource_id for candidate in context.route_candidates}
        return tuple(OptimizationResource(resource_id=resource_id) for resource_id in resource_ids)

    @staticmethod
    def _to_route_options(context: BaselinePlanScoringContext) -> tuple[OptimizationRouteOption, ...]:
        options = []
        for candidate in context.route_candidates:
            is_reachable = candidate.status == "reachable"
            options.append(
                OptimizationRouteOption(
                    route_result_id=candidate.route_result_id,
                    resource_id=candidate.resource_id,
                    response_target_id=candidate.response_target_id,
                    is_reachable=is_reachable,
                    travel_time_seconds=candidate.travel_time_seconds if is_reachable else None,
                    distance_meters=candidate.distance_meters if is_reachable else None,
                )
            )
        return tuple(options)
