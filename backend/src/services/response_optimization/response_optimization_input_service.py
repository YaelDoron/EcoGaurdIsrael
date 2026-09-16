"""Build US 5.2 optimization input from persisted US 5.1 routing snapshots."""
from __future__ import annotations

from src.models.optimization_resource import OptimizationResource
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.routing import RouteStatus
from src.repositories.response_target_repository import ResponseTargetRepository, StoredResponseTargetSet
from src.repositories.route_planning_repository import RoutePlanningRepository, StoredRoutePlanningRun


class ResponseOptimizationInputServiceError(ValueError):
    """Raised when persisted routing/target snapshots cannot form optimization input."""


class ResponseOptimizationInputService:
    """Adapter from exact RoutePlanningRun persistence to ResponseOptimizationInput."""

    def __init__(
        self,
        *,
        route_planning_repository: RoutePlanningRepository | None = None,
        response_target_repository: ResponseTargetRepository | None = None,
    ) -> None:
        self._route_planning_repository = route_planning_repository or RoutePlanningRepository()
        self._response_target_repository = response_target_repository or ResponseTargetRepository()

    def build_from_route_planning_run(self, route_planning_run_id: int) -> ResponseOptimizationInput:
        """Build optimization input from one exact persisted routing run."""
        self._validate_positive_int("route_planning_run_id", route_planning_run_id)

        stored_run = self._route_planning_repository.get_by_id(route_planning_run_id)
        if stored_run is None:
            raise ResponseOptimizationInputServiceError(
                f"RoutePlanningRun {route_planning_run_id!r} was not found."
            )

        stored_target_set = self._response_target_repository.get_by_id(stored_run.run.response_target_set_id)
        if stored_target_set is None:
            raise ResponseOptimizationInputServiceError(
                f"ResponseTargetSet {stored_run.run.response_target_set_id!r} was not found."
            )

        self._validate_snapshot(stored_run, stored_target_set)

        targets = tuple(
            OptimizationTarget(
                response_target_id=stored_target.id,
                target_order=stored_target.target_order,
                target_type=stored_target.target.target_type,
                priority_score=stored_target.target.priority_score,
            )
            for stored_target in stored_target_set.targets
        )
        resources = tuple(OptimizationResource(resource_id=resource_id) for resource_id in stored_run.run.resource_ids)
        route_options = tuple(
            OptimizationRouteOption(
                route_result_id=stored_route.id,
                resource_id=stored_route.route_result.resource_id,
                response_target_id=stored_route.route_result.response_target_id,
                is_reachable=stored_route.route_result.status == RouteStatus.REACHABLE,
                travel_time_seconds=(
                    stored_route.route_result.travel_time_seconds
                    if stored_route.route_result.status == RouteStatus.REACHABLE
                    else None
                ),
                distance_meters=(
                    stored_route.route_result.distance_meters
                    if stored_route.route_result.status == RouteStatus.REACHABLE
                    else None
                ),
            )
            for stored_route in stored_run.routes
        )

        return ResponseOptimizationInput(
            fire_event_id=stored_run.run.fire_event_id,
            response_target_set_id=stored_run.run.response_target_set_id,
            route_planning_run_id=stored_run.id,
            targets=targets,
            resources=resources,
            route_options=route_options,
        )

    @staticmethod
    def _validate_snapshot(
        stored_run: StoredRoutePlanningRun,
        stored_target_set: StoredResponseTargetSet,
    ) -> None:
        if stored_run.run.fire_event_id != stored_target_set.target_set.fire_event_id:
            raise ResponseOptimizationInputServiceError(
                "RoutePlanningRun fire_event_id must match ResponseTargetSet fire_event_id, got "
                f"{stored_run.run.fire_event_id!r} and {stored_target_set.target_set.fire_event_id!r}."
            )
        if stored_run.run.response_target_set_id != stored_target_set.id:
            raise ResponseOptimizationInputServiceError(
                "RoutePlanningRun response_target_set_id must match loaded ResponseTargetSet id, got "
                f"{stored_run.run.response_target_set_id!r} and {stored_target_set.id!r}."
            )

        resource_ids = tuple(stored_run.run.resource_ids)
        if len(resource_ids) != len(set(resource_ids)):
            raise ResponseOptimizationInputServiceError("RoutePlanningRun resource_ids must be unique.")

        target_ids = tuple(stored_target.id for stored_target in stored_target_set.targets)
        if len(target_ids) != len(set(target_ids)):
            raise ResponseOptimizationInputServiceError("ResponseTargetSet target ids must be unique.")

        valid_resources = set(resource_ids)
        valid_targets = set(target_ids)
        route_ids = []
        route_pairs = []
        for stored_route in stored_run.routes:
            route = stored_route.route_result
            route_ids.append(stored_route.id)
            route_pairs.append((route.resource_id, route.response_target_id))
            if route.resource_id not in valid_resources:
                raise ResponseOptimizationInputServiceError(
                    f"RouteResult {stored_route.id!r} references resource_id {route.resource_id!r}, "
                    "which is not in the RoutePlanningRun resource snapshot."
                )
            if route.response_target_id not in valid_targets:
                raise ResponseOptimizationInputServiceError(
                    f"RouteResult {stored_route.id!r} references response_target_id "
                    f"{route.response_target_id!r}, which is not in the ResponseTargetSet."
                )

        if len(route_ids) != len(set(route_ids)):
            raise ResponseOptimizationInputServiceError("RouteResult ids must be unique within a routing run.")
        if len(route_pairs) != len(set(route_pairs)):
            raise ResponseOptimizationInputServiceError(
                "RouteResults must not duplicate resource_id/response_target_id pairs."
            )

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResponseOptimizationInputServiceError(
                f"{field_name} must be a positive integer, got {value!r}."
            )
