"""Read-only response-plan presentation enrichment (Epic 6, US 6.3, Task 4).

`ResponsePlanPresenter` turns an existing `ResponsePlanDetails` (US 5.5) into
the fully enriched `ResponsePlanDetailResponse` the UI needs, by reading -
never recalculating - the exact persisted rows that plan already references:

    persisted Epic 5 data -> ResponsePlanDetails -> ResponsePlanPresenter -> DTO

For a given `ResponsePlanDetails`, every lookup below is keyed off IDs
already carried on that exact object (`plan_id`, `response_target_set_id`,
`route_planning_run_id`, each action's `station_id`/`response_target_id`) -
never "latest for this FireEvent", never "currently available resources".
This is what lets a superseded plan still render its own preserved planning
snapshot correctly, independent of whatever is current now.

This module performs no pathfinding, no node-mapping, no optimization, no
fitness/scoring, no baseline calculation, no planning refresh, and no
resource-status mutation - it only issues read queries against
`ResponsePlanRepository`, `ResponseTargetRepository`, `RoutePlanningRepository`,
`FireStationRepository`, and the new `GraphNodeReadRepository`, then copies
fields into presentation DTOs. Not wired into any FastAPI endpoint yet -
that is a separate task.
"""
from __future__ import annotations

from collections.abc import Iterable

from src.api.schemas.response_plans import (
    BaselineComparisonResponse,
    CoordinateResponse,
    OptimizationConfigResponse,
    ResponsePlanActionResponse,
    ResponsePlanDetailResponse,
    ResponsePlanMetricsResponse,
    ResponsePlanResourceResponse,
    ResponsePlanRouteResponse,
    ResponsePlanTargetResponse,
)
from src.database.models.fire_station_db import FireStationDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models.response_plan_details import ResponseActionDetails, ResponsePlanDetails
from src.models.response_plan_status import ResponsePlanStatus
from src.models.response_target import ResponseTarget
from src.models.routing import RouteResult
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.graph_node_read_repository import GraphNodeReadRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository


class ResponsePlanPresenter:
    """Enrich a `ResponsePlanDetails` read model into the full UI-facing DTO."""

    def __init__(
        self,
        *,
        response_plan_repository: ResponsePlanRepository | None = None,
        response_target_repository: ResponseTargetRepository | None = None,
        route_planning_repository: RoutePlanningRepository | None = None,
        fire_station_repository: FireStationRepository | None = None,
        graph_node_read_repository: GraphNodeReadRepository | None = None,
    ) -> None:
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._response_target_repository = response_target_repository or ResponseTargetRepository()
        self._route_planning_repository = route_planning_repository or RoutePlanningRepository()
        self._fire_station_repository = fire_station_repository or FireStationRepository()
        self._graph_node_read_repository = graph_node_read_repository or GraphNodeReadRepository()

    def present(self, details: ResponsePlanDetails) -> ResponsePlanDetailResponse:
        """Build the complete enriched DTO for one `ResponsePlanDetails`."""
        status = self._resolve_status(details.plan_id)
        targets_by_id = self._load_targets_by_id(details.response_target_set_id)
        routes_by_key, resource_ids = self._load_routing_run(details.route_planning_run_id)
        stations_by_id = self._load_stations_by_id()
        nodes_by_id = self._graph_node_read_repository.get_by_ids(
            self._collect_node_ids(details.actions, routes_by_key)
        )

        return ResponsePlanDetailResponse(
            plan_id=details.plan_id,
            fire_event_id=details.fire_event_id,
            response_target_set_id=details.response_target_set_id,
            route_planning_run_id=details.route_planning_run_id,
            generated_at=details.generated_at,
            methodology=details.methodology,
            methodology_version=details.methodology_version,
            random_seed=details.random_seed,
            status=status,
            is_current=details.is_current,
            metrics=ResponsePlanMetricsResponse(
                plan_score=details.plan_score,
                coverage_score=details.coverage_score,
                average_eta_seconds=details.average_eta_seconds,
            ),
            actions=[
                self._present_action(action, targets_by_id, routes_by_key, stations_by_id, nodes_by_id)
                for action in details.actions
            ],
            uncovered_targets=[
                self._present_target(target_id, targets_by_id) for target_id in details.uncovered_target_ids
            ],
            baseline_comparison=(
                BaselineComparisonResponse(
                    baseline_score=details.baseline_comparison.baseline_score,
                    baseline_coverage_score=details.baseline_comparison.baseline_coverage_score,
                    baseline_average_eta_seconds=details.baseline_comparison.baseline_average_eta_seconds,
                    score_difference=details.baseline_comparison.score_difference,
                    improvement_percentage=details.baseline_comparison.improvement_percentage,
                )
                if details.baseline_comparison is not None
                else None
            ),
            optimization_config=(
                OptimizationConfigResponse(
                    population_size=details.optimization_config.population_size,
                    generation_count=details.optimization_config.generation_count,
                    mutation_rate=details.optimization_config.mutation_rate,
                    crossover_rate=details.optimization_config.crossover_rate,
                    eta_reference_seconds=details.optimization_config.eta_reference_seconds,
                    initial_assignment_probability=details.optimization_config.initial_assignment_probability,
                    tournament_size=details.optimization_config.tournament_size,
                    elitism_count=details.optimization_config.elitism_count,
                )
                if details.optimization_config is not None
                else None
            ),
            no_resources_during_planning=len(resource_ids) == 0,
        )

    def _resolve_status(self, plan_id: int) -> ResponsePlanStatus:
        stored_plan = self._response_plan_repository.get_by_id(plan_id)
        if stored_plan is None:
            raise ValueError(f"ResponsePlan {plan_id} referenced by ResponsePlanDetails was not found.")
        return stored_plan.plan.status

    def _load_targets_by_id(self, response_target_set_id: int) -> dict[int, ResponseTarget]:
        stored_target_set = self._response_target_repository.get_by_id(response_target_set_id)
        if stored_target_set is None:
            return {}
        return {stored_target.id: stored_target.target for stored_target in stored_target_set.targets}

    def _load_routing_run(self, route_planning_run_id: int) -> tuple[dict[tuple[str, int], RouteResult], tuple]:
        stored_run = self._route_planning_repository.get_by_id(route_planning_run_id)
        if stored_run is None:
            return {}, ()
        routes_by_key = {
            (str(route.resource_id), route.response_target_id): route for route in stored_run.run.routes
        }
        return routes_by_key, stored_run.run.resource_ids

    def _load_stations_by_id(self) -> dict[str, FireStationDB]:
        return {station.id: station for station in self._fire_station_repository.get_all_stations()}

    @staticmethod
    def _collect_node_ids(
        actions: Iterable[ResponseActionDetails],
        routes_by_key: dict[tuple[str, int], RouteResult],
    ) -> set[int]:
        node_ids: set[int] = set()
        for action in actions:
            route = routes_by_key.get((str(action.resource_id), action.response_target_id))
            if route is not None:
                node_ids.update(route.node_path)
        return node_ids

    def _present_action(
        self,
        action: ResponseActionDetails,
        targets_by_id: dict[int, ResponseTarget],
        routes_by_key: dict[tuple[str, int], RouteResult],
        stations_by_id: dict[str, FireStationDB],
        nodes_by_id: dict[int, GraphNodeDB],
    ) -> ResponsePlanActionResponse:
        return ResponsePlanActionResponse(
            resource=self._present_resource(action, stations_by_id),
            target=self._present_target(action.response_target_id, targets_by_id),
            route=self._present_route(action, routes_by_key, nodes_by_id),
        )

    @staticmethod
    def _present_resource(
        action: ResponseActionDetails,
        stations_by_id: dict[str, FireStationDB],
    ) -> ResponsePlanResourceResponse:
        station = stations_by_id.get(action.station_id)
        return ResponsePlanResourceResponse(
            resource_id=action.resource_id,
            station_id=action.station_id,
            station_name=station.name if station is not None else None,
            origin=(
                CoordinateResponse(latitude=station.latitude, longitude=station.longitude)
                if station is not None
                else None
            ),
        )

    @staticmethod
    def _present_target(
        response_target_id: int,
        targets_by_id: dict[int, ResponseTarget],
    ) -> ResponsePlanTargetResponse:
        target = targets_by_id.get(response_target_id)
        if target is None:
            return ResponsePlanTargetResponse(
                response_target_id=response_target_id,
                target_type=None,
                priority_score=None,
                latitude=None,
                longitude=None,
            )
        return ResponsePlanTargetResponse(
            response_target_id=response_target_id,
            target_type=target.target_type.value,
            priority_score=target.priority_score,
            latitude=target.latitude,
            longitude=target.longitude,
        )

    def _present_route(
        self,
        action: ResponseActionDetails,
        routes_by_key: dict[tuple[str, int], RouteResult],
        nodes_by_id: dict[int, GraphNodeDB],
    ) -> ResponsePlanRouteResponse:
        route = routes_by_key.get((str(action.resource_id), action.response_target_id))
        if route is None:
            return ResponsePlanRouteResponse(
                status=None,
                eta_seconds=None,
                distance_meters=None,
                node_path=None,
                path_coordinates=None,
            )
        return ResponsePlanRouteResponse(
            status=route.status,
            eta_seconds=route.travel_time_seconds,
            distance_meters=route.distance_meters,
            node_path=list(route.node_path) if route.node_path else None,
            path_coordinates=self._resolve_path_coordinates(route.node_path, nodes_by_id),
        )

    @staticmethod
    def _resolve_path_coordinates(
        node_path: tuple[int, ...],
        nodes_by_id: dict[int, GraphNodeDB],
    ) -> list[CoordinateResponse] | None:
        if not node_path:
            return None
        coordinates: list[CoordinateResponse] = []
        for node_id in node_path:
            node = nodes_by_id.get(node_id)
            if node is None:
                return None
            coordinates.append(CoordinateResponse(latitude=node.latitude, longitude=node.longitude))
        return coordinates
