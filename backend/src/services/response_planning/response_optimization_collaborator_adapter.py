"""Task 6 adapter bridging Task 5's OptimizationCollaborator port to Company 2's ResponseOptimizationAgent.

ResponseOptimizationAgent.optimize_from_input(...) expects an already-built
ResponseOptimizationInput; the frozen Task 5 contract calls with
(route_planning_run_id, as_of, seed) instead. This adapter reconstructs the
input EXCLUSIVELY from the EXACT persisted RoutePlanningRun snapshot
identified by route_planning_run_id:

- fire_event_id / response_target_set_id / resource_ids / stored RouteResults
  come from that RoutePlanningRun row only (RoutePlanningRepository.get_by_id),
  never from current AVAILABLE resources or newly computed routes.
- targets come from the EXACT ResponseTargetSet the run references
  (ResponseTargetRepository.get_by_id(run.response_target_set_id)), never
  from "latest ResponseTargetSet".

No Dijkstra, no GA, no resource-availability or target-priority calculation
happens here - this module only translates an already-persisted snapshot
into Company 2's existing input shape and delegates exactly once.
"""
from __future__ import annotations

from datetime import datetime

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.analysis.response_optimization_result import ResponseOptimizationResult
from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.models.optimization_resource import OptimizationResource
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.routing import RouteStatus
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository


class ResponseOptimizationAdapterError(Exception):
    """Raised when the exact routing snapshot needed for optimization cannot be reconstructed."""


class ResponseOptimizationCollaboratorAdapter:
    """Bridges Task 5's OptimizationCollaborator port to the real ResponseOptimizationAgent."""

    def __init__(
        self,
        optimization_agent: ResponseOptimizationAgent,
        route_planning_repository: RoutePlanningRepository | None = None,
        response_target_repository: ResponseTargetRepository | None = None,
    ) -> None:
        self._optimization_agent = optimization_agent
        self._route_planning_repository = route_planning_repository or RoutePlanningRepository()
        self._response_target_repository = response_target_repository or ResponseTargetRepository()

    def optimize(
        self,
        *,
        route_planning_run_id: int,
        as_of: datetime,
        seed: int,
    ) -> ResponseOptimizationResult:
        """Optimize the EXACT persisted routing snapshot identified by route_planning_run_id."""
        self._validate_request(route_planning_run_id, as_of, seed)

        stored_run = self._route_planning_repository.get_by_id(route_planning_run_id)
        if stored_run is None:
            raise ResponseOptimizationAdapterError(f"RoutePlanningRun {route_planning_run_id!r} was not found.")

        stored_target_set = self._response_target_repository.get_by_id(stored_run.run.response_target_set_id)
        if stored_target_set is None:
            raise ResponseOptimizationAdapterError(
                f"ResponseTargetSet {stored_run.run.response_target_set_id!r} referenced by "
                f"RoutePlanningRun {route_planning_run_id!r} was not found."
            )
        if stored_target_set.target_set.fire_event_id != stored_run.run.fire_event_id:
            raise ResponseOptimizationAdapterError(
                "ResponseTargetSet fire_event_id does not match RoutePlanningRun fire_event_id: "
                f"target_set={stored_target_set.target_set.fire_event_id!r} "
                f"run={stored_run.run.fire_event_id!r}."
            )

        optimization_input = ResponseOptimizationInput(
            fire_event_id=stored_run.run.fire_event_id,
            response_target_set_id=stored_run.run.response_target_set_id,
            route_planning_run_id=stored_run.id,
            targets=self._to_optimization_targets(stored_target_set),
            resources=self._to_optimization_resources(stored_run),
            route_options=self._to_route_options(stored_run),
        )

        config = ResponseOptimizationConfig(random_seed=seed)
        return self._optimization_agent.optimize_from_input(optimization_input, as_of=as_of, config=config)

    @staticmethod
    def _to_optimization_targets(stored_target_set) -> tuple[OptimizationTarget, ...]:
        return tuple(
            OptimizationTarget(
                response_target_id=stored_target.id,
                target_order=stored_target.target_order,
                target_type=stored_target.target.target_type,
                priority_score=stored_target.target.priority_score,
            )
            for stored_target in stored_target_set.targets
        )

    @staticmethod
    def _to_optimization_resources(stored_run) -> tuple[OptimizationResource, ...]:
        # Membership comes exclusively from the run's own persisted
        # resource_ids snapshot - never from a current-availability query.
        return tuple(OptimizationResource(resource_id=resource_id) for resource_id in stored_run.run.resource_ids)

    @staticmethod
    def _to_route_options(stored_run) -> tuple[OptimizationRouteOption, ...]:
        # Stored routing output is authoritative: no rerouting, no ETA/distance
        # recalculation - every stored RouteResult is mapped through unchanged.
        options = []
        for stored_route in stored_run.routes:
            route = stored_route.route_result
            is_reachable = route.status is RouteStatus.REACHABLE
            options.append(
                OptimizationRouteOption(
                    route_result_id=stored_route.id,
                    resource_id=route.resource_id,
                    response_target_id=route.response_target_id,
                    is_reachable=is_reachable,
                    travel_time_seconds=route.travel_time_seconds if is_reachable else None,
                    distance_meters=route.distance_meters if is_reachable else None,
                )
            )
        return tuple(options)

    @staticmethod
    def _validate_request(route_planning_run_id: object, as_of: object, seed: object) -> None:
        if (
            isinstance(route_planning_run_id, bool)
            or not isinstance(route_planning_run_id, int)
            or route_planning_run_id <= 0
        ):
            raise ValueError(f"route_planning_run_id must be a positive integer, got {route_planning_run_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
