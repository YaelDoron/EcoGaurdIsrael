"""Structural dependency ports for `BaselineComparisonService` (US 5.3 Task 5).

`BaselineComparisonService` needs to load an exact persisted optimized
plan, the exact `RoutePlanningRun` it used, and the exact
`ResponseTargetSet` it used. As of Task 5, Company 1's `RoutePlanningRun`/
`RouteResult` persistence and Company 2's `ResponsePlan` persistence still
do not exist in this codebase (re-checked -- see
`baseline_comparison_service.py`). Rather than fabricate production copies
of those tables/repositories, this module defines minimal structural
`Protocol` ports describing only the fields/methods Task 5 actually reads.

These are dependency contracts, not replacement repositories:
- `OptimizedPlanReader` / `RoutePlanningRunReader` need real adapters once
  Company 1/2 merge their persistence -- any object whose `get_by_id`
  returns something shaped like `OptimizedPlanLike`/`RoutePlanningRunLike`
  satisfies the port structurally, no inheritance required.
- `ResponseTargetSetReader` is already satisfied by the real, existing US
  4.3 `ResponseTargetRepository.get_by_id(...) -> StoredResponseTargetSet
  | None` -- no adapter needed, it is used directly in production.

`RouteResultLike.status` reuses Task 1's exact `RouteCandidateStatus`
literal and `OptimizedPlanLike.score` reuses Task 2's exact
`PlanScoreBreakdownLike` -- nothing here invents a competing RouteStatus
or PlanScoreBreakdown shape.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from src.calculators.baseline_plan.baseline_plan_calculator import RouteCandidateStatus
from src.calculators.baseline_plan.baseline_plan_evaluator import PlanScoreBreakdownLike
from src.repositories.response_target_repository import StoredResponseTargetSet


class RouteResultLike(Protocol):
    """Structural shape of one persisted route result inside a RoutePlanningRun snapshot."""

    id: int
    resource_id: str
    response_target_id: int
    status: RouteCandidateStatus
    travel_time_seconds: float | None
    distance_meters: float | None


class RoutePlanningRunLike(Protocol):
    """Structural shape of the exact persisted routing snapshot used by one optimized plan."""

    id: int
    fire_event_id: int
    response_target_set_id: int
    resource_ids: Sequence[str]
    route_results: Sequence[RouteResultLike]


class OptimizedPlanLike(Protocol):
    """Structural shape of the exact persisted optimized plan Task 5 reads.

    `score` reuses Task 2's `PlanScoreBreakdownLike` -- Task 5 never
    recomputes it, only copies it into `OptimizedPlanEvaluation`.
    """

    id: int
    fire_event_id: int
    route_planning_run_id: int
    response_target_set_id: int
    score: PlanScoreBreakdownLike


class OptimizedPlanReader(Protocol):
    """Port for loading one exact persisted optimized ResponsePlan by id."""

    def get_by_id(self, response_plan_id: int) -> OptimizedPlanLike | None: ...


class RoutePlanningRunReader(Protocol):
    """Port for loading one exact persisted RoutePlanningRun by id."""

    def get_by_id(self, route_planning_run_id: int) -> RoutePlanningRunLike | None: ...


class ResponseTargetSetReader(Protocol):
    """Port for loading one exact persisted ResponseTargetSet by id.

    Structurally satisfied today by the real `ResponseTargetRepository`.
    """

    def get_by_id(self, response_target_set_id: int) -> StoredResponseTargetSet | None: ...
