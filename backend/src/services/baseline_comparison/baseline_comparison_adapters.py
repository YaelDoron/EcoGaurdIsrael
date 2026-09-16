"""Real repository and scorer adapters for US 5.3 baseline comparison."""
from __future__ import annotations

from dataclasses import dataclass

from src.calculators.baseline_plan.baseline_plan_evaluator import (
    BaselinePlanScorer,
    BaselinePlanScoringContext,
)
from src.calculators.response_optimization import ResponsePlanScorer
from src.models.response_action import ResponseAction
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.response_optimization import ResponseOptimizationInputService


@dataclass(frozen=True)
class PersistedOptimizedPlanScore:
    """Persisted optimized metrics exposed through the US 5.3 score port."""

    total_score: float
    coverage_score: float
    average_eta_seconds: float | None
    covered_target_count: int
    total_target_count: int
    uncovered_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class PersistedOptimizedPlanSnapshot:
    """Exact persisted ResponsePlan snapshot needed by US 5.3."""

    id: int
    fire_event_id: int
    route_planning_run_id: int
    response_target_set_id: int
    score: PersistedOptimizedPlanScore


@dataclass(frozen=True)
class PersistedRouteResultSnapshot:
    """Exact persisted RouteResult fields needed by the greedy baseline."""

    id: int
    resource_id: str
    response_target_id: int
    status: str
    travel_time_seconds: float | None


@dataclass(frozen=True)
class PersistedRoutePlanningRunSnapshot:
    """Exact persisted RoutePlanningRun snapshot needed by US 5.3."""

    id: int
    fire_event_id: int
    response_target_set_id: int
    resource_ids: tuple[str, ...]
    route_results: tuple[PersistedRouteResultSnapshot, ...]


class ResponsePlanRepositoryOptimizedPlanReader:
    """Adapter from the real US 5.2 ResponsePlanRepository to the US 5.3 port."""

    def __init__(self, repository: ResponsePlanRepository | None = None) -> None:
        self._repository = repository or ResponsePlanRepository()

    def get_by_id(self, response_plan_id: int) -> PersistedOptimizedPlanSnapshot | None:
        stored_plan = self._repository.get_by_id(response_plan_id)
        if stored_plan is None:
            return None

        plan = stored_plan.plan
        if plan.plan_score is None or plan.coverage_score is None:
            raise ValueError(
                f"ResponsePlan {response_plan_id!r} is missing persisted optimized score metrics."
            )

        return PersistedOptimizedPlanSnapshot(
            id=stored_plan.id,
            fire_event_id=plan.fire_event_id,
            route_planning_run_id=plan.route_planning_run_id,
            response_target_set_id=plan.response_target_set_id,
            score=PersistedOptimizedPlanScore(
                total_score=plan.plan_score,
                coverage_score=plan.coverage_score,
                average_eta_seconds=plan.average_eta_seconds,
                covered_target_count=len(plan.actions),
                total_target_count=len(plan.actions) + len(plan.uncovered_target_ids),
                uncovered_target_ids=plan.uncovered_target_ids,
            ),
        )


class RoutePlanningRepositoryRunReader:
    """Adapter from the real US 5.1 RoutePlanningRepository to the US 5.3 port."""

    def __init__(self, repository: RoutePlanningRepository | None = None) -> None:
        self._repository = repository or RoutePlanningRepository()

    def get_by_id(self, route_planning_run_id: int) -> PersistedRoutePlanningRunSnapshot | None:
        stored_run = self._repository.get_by_id(route_planning_run_id)
        if stored_run is None:
            return None

        return PersistedRoutePlanningRunSnapshot(
            id=stored_run.id,
            fire_event_id=stored_run.run.fire_event_id,
            response_target_set_id=stored_run.run.response_target_set_id,
            resource_ids=stored_run.run.resource_ids,
            route_results=tuple(
                PersistedRouteResultSnapshot(
                    id=stored_route.id,
                    resource_id=stored_route.route_result.resource_id,
                    response_target_id=stored_route.route_result.response_target_id,
                    status=stored_route.route_result.status.value,
                    travel_time_seconds=stored_route.route_result.travel_time_seconds,
                )
                for stored_route in stored_run.routes
            ),
        )


class ResponsePlanScorerBaselineAdapter(BaselinePlanScorer):
    """Scores the baseline allocation with the real US 5.2 ResponsePlanScorer."""

    def __init__(
        self,
        *,
        input_service: ResponseOptimizationInputService | None = None,
        scorer: ResponsePlanScorer | None = None,
    ) -> None:
        self._input_service = input_service or ResponseOptimizationInputService()
        self._scorer = scorer or ResponsePlanScorer()

    def evaluate(self, context: BaselinePlanScoringContext):
        optimization_input = self._input_service.build_from_route_planning_run(context.route_planning_run_id)
        if optimization_input.response_target_set_id != context.response_target_set_id:
            raise ValueError(
                "ResponseOptimizationInput response_target_set_id does not match baseline context, got "
                f"{optimization_input.response_target_set_id!r} and {context.response_target_set_id!r}."
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
