"""GlobalRouteMatrixBuilder: the genuine cross-event route matrix (Stage 3
of the Global Multi-Incident Optimizer refactor, Tasks 10-16, 22).

Computes routes for the full cross product of {assignable resources} x
{all targets from every active event}, NOT per-event subsets - a resource
originally gathered near FireEvent A's local area is still routed to
FireEvent B's targets if geographically feasible (Task 16/22). Reuses
NodeMappingService/DijkstraCalculator UNCHANGED (Task 1/12) - no new
pathfinding logic. UNAVAILABLE resources (Task 4/22's one exclusion rule)
are skipped entirely; a route is never computed for them.

Deduplication (Task 14): resources at the same station map to the same
source_node_id, so the (source_node_id, target_node_id) DijkstraResult is
cached and reused across every resource sharing that origin, without
touching DijkstraCalculator's own algorithm.

Unreachable/unmappable pairs (Task 15) simply produce no GlobalRouteOption
- never a failure of the whole build.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.calculators.routing.dijkstra_calculator import DijkstraCalculator, DijkstraResult
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.routing import RouteStatus, RoutingResource, RoutingTarget
from src.services.routing.node_mapping_service import NodeMappingService


@dataclass(frozen=True)
class GlobalRouteMatrixBuildResult:
    """The matrix plus Task 30's search-space instrumentation for one build."""

    matrix: GlobalRouteMatrix
    resource_count: int
    target_count: int
    potential_pairs: int
    feasible_pairs: int
    dijkstra_call_count: int


class GlobalRouteMatrixBuilder:
    """Builds one cross-event GlobalRouteMatrix from a resource universe and a target set."""

    def __init__(
        self,
        node_mapping_service: NodeMappingService | None = None,
        dijkstra_calculator: DijkstraCalculator | None = None,
    ) -> None:
        self._node_mapping_service = node_mapping_service or NodeMappingService()
        self._dijkstra_calculator = dijkstra_calculator or DijkstraCalculator()

    def build(
        self,
        resources: tuple[GlobalPlanningResource, ...],
        targets: tuple[GlobalPlanningTarget, ...],
        road_nodes: list[GraphNode],
        road_edges: list[GraphEdge],
    ) -> GlobalRouteMatrixBuildResult:
        assignable_resources = tuple(resource for resource in resources if resource.is_assignable)
        potential_pairs = len(assignable_resources) * len(targets)

        if not assignable_resources or not targets:
            return GlobalRouteMatrixBuildResult(
                matrix=GlobalRouteMatrix(options=()),
                resource_count=len(assignable_resources),
                target_count=len(targets),
                potential_pairs=potential_pairs,
                feasible_pairs=0,
                dijkstra_call_count=0,
            )

        routing_resources = [
            RoutingResource(
                resource_id=resource.resource_id,
                station_id=resource.station_id,
                latitude=resource.station_latitude,
                longitude=resource.station_longitude,
            )
            for resource in assignable_resources
        ]
        routing_targets = [
            RoutingTarget(
                response_target_id=target.response_target_id,
                target_order=target.target_order,
                target_type=target.target_type.value,
                latitude=target.latitude,
                longitude=target.longitude,
                priority_score=target.priority_score,
            )
            for target in targets
        ]

        resource_node_map = self._node_mapping_service.map_resources(routing_resources, road_nodes, road_edges)
        target_node_map = self._node_mapping_service.map_targets(routing_targets, road_nodes, road_edges)

        # Optimization 1: one adjacency-list build for the whole matrix,
        # not one per (source, target) search - road_edges never changes
        # across the loop below.
        routing_graph = self._dijkstra_calculator.build_graph(road_edges)

        dijkstra_cache: dict[tuple[int, int], DijkstraResult] = {}
        dijkstra_call_count = 0
        options: list[GlobalRouteOption] = []

        for resource in assignable_resources:
            source_node_id = resource_node_map.get(resource.resource_id)
            if source_node_id is None:
                continue
            for target in targets:
                target_node_id = target_node_map.get(target.response_target_id)
                if target_node_id is None:
                    continue

                cache_key = (source_node_id, target_node_id)
                cached_result = dijkstra_cache.get(cache_key)
                if cached_result is None:
                    cached_result = self._dijkstra_calculator.calculate_shortest_path_in_graph(
                        source_node_id, target_node_id, routing_graph
                    )
                    dijkstra_cache[cache_key] = cached_result
                    dijkstra_call_count += 1

                if cached_result.status is not RouteStatus.REACHABLE:
                    continue

                options.append(
                    GlobalRouteOption(
                        resource_id=resource.resource_id,
                        fire_event_id=target.fire_event_id,
                        response_target_id=target.response_target_id,
                        eta_seconds=cached_result.travel_time_seconds,
                        route_distance_meters=cached_result.distance_meters,
                        node_path=cached_result.node_path,
                    )
                )

        matrix = GlobalRouteMatrix(options=tuple(options))
        return GlobalRouteMatrixBuildResult(
            matrix=matrix,
            resource_count=len(assignable_resources),
            target_count=len(targets),
            potential_pairs=potential_pairs,
            feasible_pairs=len(matrix),
            dijkstra_call_count=dijkstra_call_count,
        )
