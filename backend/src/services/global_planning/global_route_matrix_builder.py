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

Unmappable pairs (Task 15) simply produce no GlobalRouteOption - never a
failure of the whole build.

Disconnected-graph fallback: a pair whose endpoints both map to a node but
between which Dijkstra finds NO path (UNREACHABLE - a fetched graph split
into disconnected components) is no longer dropped, which used to leave
fires with no response actions. It becomes a GlobalRouteOption whose ETA is
the station->target straight-line (Haversine) distance at
DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH and whose node_path is just
(source_node_id, target_node_id) - a straight segment, never a routed path.
Only within MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM: a station farther
from the target than that stays infeasible, so a split graph can never turn
a station in another region into a candidate for a far-away fire.
`GlobalRouteMatrixBuildResult.fallback_pair_count` reports how many such
estimated pairs a build contained. UNMAPPABLE pairs still get no option:
there is no node to build even that two-node path from.

First-Mile Heuristic Fallback (see _first_mile_penalty): even with
GlobalPlanningInputBuilder's retrying, persistent-cache-backed station-micro
fetch (Step 4), Overpass can still exhaust every retry for a given station in
a given cycle, leaving NodeMappingService no choice but to snap that
resource's origin to whatever real node IS in the fetched graph - which can
be a real but non-trivial distance from the station's true coordinate. This
first-mile correction never invents a route, never runs when Dijkstra
found no REACHABLE path, and never replaces a single meter of the real,
road-network-derived path Dijkstra returns. It only adds a small, bounded
correction on TOP OF an already-real Dijkstra route, honestly reflecting the
one un-routable stretch (station door -> nearest fetched road node) that no
road-network fetch can ever eliminate from ANY routing system, real or
simulated - the same reason Google/Waze-class routers always add their own
short "walk/drive to the road" estimate for an origin that isn't already
sitting on a mapped road. That penalty is separate from the disconnected-
graph fallback above, which replaces a MISSING route entirely and therefore
deliberately does not also add the first-mile penalty (its straight line
already starts at the station's true coordinate).

FIRST_MILE_PENALTY_THRESHOLD_KM lives in src/utils/geo.py, not here - it is
shared with ResponsePlanPresenter, which bridges this same gap visually in
`path_coordinates` (one honest straight-line segment from the station's real
coordinate, never touching node_path itself) using the identical threshold,
so the two layers can never independently drift out of agreement about
which routes need a correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

from src.calculators.routing.dijkstra_calculator import DijkstraCalculator, DijkstraResult
from src.calculators.routing.haversine_fallback_calculator import (
    DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH,
    estimate_disconnected_graph_distance_and_eta,
)
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.routing import RouteStatus, RoutingResource, RoutingTarget
from src.services.routing.node_mapping_service import NodeMappingService
from src.utils.geo import FIRST_MILE_PENALTY_THRESHOLD_KM, haversine_distance_km

# Conservative (i.e. SLOW) on purpose: a real first/last-mile stretch (a
# station forecourt, a short unclassified access road) is driven far below
# highway speed, and Dijkstra's own edge speeds already handle everything
# ONCE a vehicle reaches a mapped road - this constant only ever covers the
# one un-routable stretch outside the fetched graph. Erring slow means this
# never UNDERSTATES the true cost of a real gap.
FIRST_MILE_FALLBACK_SPEED_KMH = 30.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalRouteMatrixBuildResult:
    """The matrix plus Task 30's search-space instrumentation for one build."""

    matrix: GlobalRouteMatrix
    resource_count: int
    target_count: int
    potential_pairs: int
    feasible_pairs: int
    dijkstra_call_count: int
    # Pairs Dijkstra found unreachable that were kept as straight-line
    # estimates (see this module's docstring) rather than dropped.
    fallback_pair_count: int = 0


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
        nodes_by_id = {node.id: node for node in road_nodes}

        # Optimization 1: one adjacency-list build for the whole matrix,
        # not one per (source, target) search - road_edges never changes
        # across the loop below.
        routing_graph = self._dijkstra_calculator.build_graph(road_edges)

        dijkstra_cache: dict[tuple[int, int], DijkstraResult] = {}
        dijkstra_call_count = 0
        fallback_pair_count = 0
        options: list[GlobalRouteOption] = []

        for resource in assignable_resources:
            source_node_id = resource_node_map.get(resource.resource_id)
            if source_node_id is None:
                continue
            extra_distance_meters, extra_seconds = self._first_mile_penalty(
                resource.station_latitude, resource.station_longitude, nodes_by_id.get(source_node_id)
            )
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
                    fallback_option = self._disconnected_graph_fallback_option(
                        resource, target, source_node_id, target_node_id
                    )
                    if fallback_option is not None:
                        options.append(fallback_option)
                        fallback_pair_count += 1
                    continue

                options.append(
                    GlobalRouteOption(
                        resource_id=resource.resource_id,
                        fire_event_id=target.fire_event_id,
                        response_target_id=target.response_target_id,
                        eta_seconds=cached_result.travel_time_seconds + extra_seconds,
                        route_distance_meters=cached_result.distance_meters + extra_distance_meters,
                        node_path=cached_result.node_path,
                    )
                )

        if fallback_pair_count:
            logger.warning(
                "Global route matrix: %d resource/target pair(s) had no road path in the fetched graph "
                "(disconnected components); using straight-line estimates at %.0f km/h for them.",
                fallback_pair_count,
                DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH,
            )

        matrix = GlobalRouteMatrix(options=tuple(options))
        return GlobalRouteMatrixBuildResult(
            matrix=matrix,
            resource_count=len(assignable_resources),
            target_count=len(targets),
            potential_pairs=potential_pairs,
            feasible_pairs=len(matrix),
            dijkstra_call_count=dijkstra_call_count,
            fallback_pair_count=fallback_pair_count,
        )

    @staticmethod
    def _disconnected_graph_fallback_option(
        resource: GlobalPlanningResource,
        target: GlobalPlanningTarget,
        source_node_id: int,
        target_node_id: int,
    ) -> GlobalRouteOption | None:
        """A straight-line stand-in for a pair Dijkstra found no path for:
        the station->target Haversine distance/ETA, with node_path just the
        two already-mapped endpoint nodes (a straight segment for the map).
        None (the pair stays infeasible) when the station and target are
        farther apart than MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM."""
        estimate = estimate_disconnected_graph_distance_and_eta(
            resource.station_latitude, resource.station_longitude, target.latitude, target.longitude
        )
        if estimate is None:
            return None
        distance_meters, eta_seconds = estimate
        return GlobalRouteOption(
            resource_id=resource.resource_id,
            fire_event_id=target.fire_event_id,
            response_target_id=target.response_target_id,
            eta_seconds=eta_seconds,
            route_distance_meters=distance_meters,
            node_path=(source_node_id, target_node_id),
        )

    @staticmethod
    def _first_mile_penalty(
        station_latitude: float, station_longitude: float, source_node: GraphNode | None
    ) -> tuple[float, float]:
        """First-Mile Heuristic Fallback (see this module's own docstring):
        (extra_distance_meters, extra_seconds) to add on top of an already-
        REACHABLE Dijkstra route, honestly reflecting the un-routable gap
        between a station's true coordinate and the real road node
        NodeMappingService had to snap it to. (0.0, 0.0) - no correction -
        when `source_node` is missing (defensive only; every REACHABLE
        result's source_node_id is guaranteed present in road_nodes) or the
        gap is within FIRST_MILE_PENALTY_THRESHOLD_KM, the ordinary slack
        between a station's exact coordinate and its nearest real road node.

        Computed once per resource (identical for every target it routes
        to, since it depends only on the resource's own fixed station
        coordinate and its one snapped source node) - never per Dijkstra
        call, and never for a resource NodeMappingService could not map at
        all (that stays UNMAPPABLE, exactly as before; this never manufactures
        a route where none exists)."""
        if source_node is None:
            return 0.0, 0.0
        gap_km = haversine_distance_km(station_latitude, station_longitude, source_node.latitude, source_node.longitude)
        if gap_km <= FIRST_MILE_PENALTY_THRESHOLD_KM:
            return 0.0, 0.0
        extra_seconds = (gap_km / FIRST_MILE_FALLBACK_SPEED_KMH) * 3600.0
        return gap_km * 1000.0, extra_seconds
