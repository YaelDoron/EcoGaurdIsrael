"""Read-only response-plan-details aggregation service (Epic 5, US 5.5, Task 2).

ResponsePlanDetailsService assembles US 5.5's `ResponsePlanDetails` read
model purely by fetching and joining data that US 5.1 (routing), US 5.2
(optimization), and US 5.3 (baseline comparison) already persisted. It
contains no Dijkstra, no genetic-algorithm, no target-priority, and no
node-mapping logic of its own, and it never writes to any repository or
mutates `FirefightingResource.status` - it is a strict read/aggregate/map
step, matching `CurrentResponsePlanResolver`'s own read-only precedent.

Missing upstream data is handled by omission, not by raising: a response
action whose resource or response target can no longer be resolved is
dropped from the assembled `actions` tuple (and logged), a route that was
never computed leaves that action's ETA/distance/path fields `None`, and a
FireEvent with no baseline comparison yet yields `baseline_comparison=None`.
"""
from __future__ import annotations

import logging

from src.models.response_plan_details import (
    BaselineComparisonDetails,
    ResponseActionDetails,
    ResponsePlanDetails,
)
from src.models.response_action import ResponseAction
from src.models.response_target import ResponseTarget
from src.models.routing import RouteResult
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_repository import ResponsePlanRepository, StoredResponsePlan
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver

logger = logging.getLogger(__name__)


class ResponsePlanDetailsService:
    """Assemble `ResponsePlanDetails` read models from already-persisted planning data."""

    def __init__(
        self,
        *,
        current_response_plan_resolver: CurrentResponsePlanResolver | None = None,
        response_plan_repository: ResponsePlanRepository | None = None,
        route_planning_repository: RoutePlanningRepository | None = None,
        response_target_repository: ResponseTargetRepository | None = None,
        firefighting_resource_repository: FirefightingResourceRepository | None = None,
        fire_station_repository: FireStationRepository | None = None,
        plan_comparison_repository: PlanComparisonRepository | None = None,
    ) -> None:
        self._current_response_plan_resolver = current_response_plan_resolver or CurrentResponsePlanResolver()
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._route_planning_repository = route_planning_repository or RoutePlanningRepository()
        self._response_target_repository = response_target_repository or ResponseTargetRepository()
        self._firefighting_resource_repository = (
            firefighting_resource_repository or FirefightingResourceRepository()
        )
        self._fire_station_repository = fire_station_repository or FireStationRepository()
        self._plan_comparison_repository = plan_comparison_repository or PlanComparisonRepository()

    def get_current_plan_details(self, fire_event_id: int) -> ResponsePlanDetails | None:
        """Return the current planning-safe plan's details for a FireEvent, or None if there is none."""
        self._validate_positive_int("fire_event_id", fire_event_id)
        current_plan = self._current_response_plan_resolver.resolve(fire_event_id=fire_event_id)
        if current_plan is None:
            return None
        return self._assemble(current_plan, is_current=True)

    def get_plan_details_by_id(self, plan_id: int) -> ResponsePlanDetails | None:
        """Return one specific plan's details by database id, or None if it does not exist."""
        self._validate_positive_int("plan_id", plan_id)
        stored_plan = self._response_plan_repository.get_by_id(plan_id)
        if stored_plan is None:
            return None

        current_plan = self._current_response_plan_resolver.resolve(fire_event_id=stored_plan.plan.fire_event_id)
        is_current = current_plan is not None and current_plan.id == stored_plan.id
        return self._assemble(stored_plan, is_current=is_current)

    def _assemble(self, stored_plan: StoredResponsePlan, *, is_current: bool) -> ResponsePlanDetails:
        plan = stored_plan.plan

        targets_by_id = self._load_targets_by_id(plan.response_target_set_id)
        routes_by_key = self._load_routes_by_key(plan.route_planning_run_id)
        known_station_ids = self._load_known_station_ids() if plan.actions else frozenset()

        actions = tuple(
            details
            for details in (
                self._build_action_details(action, targets_by_id, routes_by_key, known_station_ids)
                for action in plan.actions
            )
            if details is not None
        )

        return ResponsePlanDetails(
            plan_id=stored_plan.id,
            fire_event_id=plan.fire_event_id,
            response_target_set_id=plan.response_target_set_id,
            route_planning_run_id=plan.route_planning_run_id,
            generated_at=plan.generated_at,
            methodology=plan.methodology,
            methodology_version=plan.methodology_version,
            is_current=is_current,
            plan_score=plan.plan_score if plan.plan_score is not None else 0.0,
            coverage_score=plan.coverage_score if plan.coverage_score is not None else 0.0,
            average_eta_seconds=plan.average_eta_seconds,
            actions=actions,
            uncovered_target_ids=plan.uncovered_target_ids,
            baseline_comparison=self._resolve_baseline_comparison(plan.fire_event_id, stored_plan.id),
        )

    def _load_targets_by_id(self, response_target_set_id: int) -> dict[int, ResponseTarget]:
        stored_target_set = self._response_target_repository.get_by_id(response_target_set_id)
        if stored_target_set is None:
            logger.warning("ResponseTargetSet %s was not found while assembling plan details", response_target_set_id)
            return {}
        return {stored_target.id: stored_target.target for stored_target in stored_target_set.targets}

    def _load_routes_by_key(self, route_planning_run_id: int) -> dict[tuple[str, int], RouteResult]:
        stored_run = self._route_planning_repository.get_by_id(route_planning_run_id)
        if stored_run is None:
            logger.warning("RoutePlanningRun %s was not found while assembling plan details", route_planning_run_id)
            return {}
        return {
            (str(route.resource_id), route.response_target_id): route
            for route in stored_run.run.routes
        }

    def _load_known_station_ids(self) -> frozenset[str]:
        return frozenset(str(station.id) for station in self._fire_station_repository.get_all_stations())

    def _build_action_details(
        self,
        action: ResponseAction,
        targets_by_id: dict[int, ResponseTarget],
        routes_by_key: dict[tuple[str, int], RouteResult],
        known_station_ids: frozenset[str],
    ) -> ResponseActionDetails | None:
        resource = self._firefighting_resource_repository.get_by_id(action.resource_id)
        if resource is None:
            logger.warning("Skipping response action for missing FirefightingResource %s", action.resource_id)
            return None

        target = targets_by_id.get(action.response_target_id)
        if target is None:
            logger.warning("Skipping response action for missing ResponseTarget %s", action.response_target_id)
            return None

        station_id = str(resource.station_id)
        if station_id not in known_station_ids:
            logger.warning("Resource %s references unknown FireStation %s", action.resource_id, station_id)

        route = routes_by_key.get((str(action.resource_id), action.response_target_id))

        return ResponseActionDetails(
            resource_id=str(action.resource_id),
            station_id=station_id,
            response_target_id=action.response_target_id,
            target_type=target.target_type.value,
            target_priority=target.priority_score,
            eta_seconds=route.travel_time_seconds if route is not None else None,
            route_distance_meters=route.distance_meters if route is not None else None,
            node_path=route.node_path if (route is not None and route.node_path) else None,
        )

    def _resolve_baseline_comparison(self, fire_event_id: int, plan_id: int) -> BaselineComparisonDetails | None:
        comparisons = self._plan_comparison_repository.list_for_fire_event(fire_event_id)
        matching = [
            stored.comparison for stored in comparisons if stored.comparison.optimized_plan_id == plan_id
        ]
        if not matching:
            return None

        comparison = matching[-1]
        return BaselineComparisonDetails(
            baseline_score=comparison.baseline_score,
            baseline_coverage_score=comparison.baseline_coverage_score,
            baseline_average_eta_seconds=comparison.baseline_average_eta_seconds,
            score_difference=comparison.score_difference,
            improvement_percentage=comparison.improvement_percentage,
        )

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
