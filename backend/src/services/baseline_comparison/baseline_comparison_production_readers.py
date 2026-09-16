"""Production adapters satisfying BaselineComparisonService's own structural ports.

`baseline_comparison_ports.py` defines `OptimizedPlanReader` and
`RoutePlanningRunReader` as pure structural `Protocol`s, written when
Company 1/2's real `RoutePlanningRun`/`ResponsePlan` persistence did not yet
exist in this codebase. Both now exist (Epic 5 Tasks preceding this one),
and every field either Protocol needs is already present on
`StoredRoutePlanningRun`/`StoredResponsePlan` - so these two adapters are
pure, read-only shape translations: no algorithm, no recalculation, no
persistence of their own.

`BaselinePlanScorer` (BaselineComparisonService's third port) is
deliberately NOT adapted here. Wiring it to the real `ResponsePlanScorer`
would require priority_score/target_type/distance_meters/fire_event_id that
US 5.3's own `TargetOrder`/`RouteCandidate`/`BaselinePlanScoringContext`
types do not carry - closing that gap means changing already-tested US 5.3
calculator files, which exceeds this integration layer's "adapters only,
smallest same-team change" mandate. Production callers must supply a real
`BaselinePlanScorer` explicitly (see response_planning_production_factory.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison.baseline_comparison_ports import OptimizedPlanLike, RoutePlanningRunLike


@dataclass(frozen=True)
class _RouteResultView:
    """Structural RouteResultLike view of one persisted RouteResult."""

    id: int
    resource_id: str
    response_target_id: int
    status: str
    travel_time_seconds: float | None
    distance_meters: float | None


@dataclass(frozen=True)
class _RoutePlanningRunView:
    """Structural RoutePlanningRunLike view of one persisted RoutePlanningRun."""

    id: int
    fire_event_id: int
    response_target_set_id: int
    resource_ids: tuple[str, ...]
    route_results: tuple[_RouteResultView, ...]


@dataclass(frozen=True)
class _PlanScoreBreakdownView:
    """Structural PlanScoreBreakdownLike view built from ResponsePlan's flat score fields."""

    total_score: float
    coverage_score: float
    average_eta_seconds: float | None
    covered_target_count: int
    total_target_count: int
    uncovered_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class _OptimizedPlanView:
    """Structural OptimizedPlanLike view of one persisted ResponsePlan."""

    id: int
    fire_event_id: int
    route_planning_run_id: int
    response_target_set_id: int
    score: _PlanScoreBreakdownView


class RoutePlanningRunReaderAdapter:
    """Satisfies BaselineComparisonService's RoutePlanningRunReader port with real persistence."""

    def __init__(self, route_planning_repository: RoutePlanningRepository | None = None) -> None:
        self._route_planning_repository = route_planning_repository or RoutePlanningRepository()

    def get_by_id(self, route_planning_run_id: int) -> RoutePlanningRunLike | None:
        stored_run = self._route_planning_repository.get_by_id(route_planning_run_id)
        if stored_run is None:
            return None
        return _RoutePlanningRunView(
            id=stored_run.id,
            fire_event_id=stored_run.run.fire_event_id,
            response_target_set_id=stored_run.run.response_target_set_id,
            resource_ids=stored_run.run.resource_ids,
            route_results=tuple(
                _RouteResultView(
                    id=stored_route.id,
                    resource_id=stored_route.route_result.resource_id,
                    response_target_id=stored_route.route_result.response_target_id,
                    status=stored_route.route_result.status.value,
                    travel_time_seconds=stored_route.route_result.travel_time_seconds,
                    distance_meters=stored_route.route_result.distance_meters,
                )
                for stored_route in stored_run.routes
            ),
        )


class ResponsePlanOptimizedPlanReaderAdapter:
    """Satisfies BaselineComparisonService's OptimizedPlanReader port with real persistence."""

    def __init__(self, response_plan_repository: ResponsePlanRepository | None = None) -> None:
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()

    def get_by_id(self, response_plan_id: int) -> OptimizedPlanLike | None:
        stored_plan = self._response_plan_repository.get_by_id(response_plan_id)
        if stored_plan is None:
            return None
        plan = stored_plan.plan
        covered_target_count = len(plan.actions)
        total_target_count = covered_target_count + len(plan.uncovered_target_ids)
        return _OptimizedPlanView(
            id=stored_plan.id,
            fire_event_id=plan.fire_event_id,
            route_planning_run_id=plan.route_planning_run_id,
            response_target_set_id=plan.response_target_set_id,
            score=_PlanScoreBreakdownView(
                total_score=plan.plan_score if plan.plan_score is not None else 0.0,
                coverage_score=plan.coverage_score if plan.coverage_score is not None else 0.0,
                average_eta_seconds=plan.average_eta_seconds,
                covered_target_count=covered_target_count,
                total_target_count=total_target_count,
                uncovered_target_ids=plan.uncovered_target_ids,
            ),
        )
