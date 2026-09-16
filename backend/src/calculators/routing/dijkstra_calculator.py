"""Pure deterministic shortest-path search over the road-network graph.

DijkstraCalculator only computes the fastest path between two already-mapped
GraphNode ids: `travel_time_seconds` is the priority-queue weight (EcoGuard
routes for speed, not physical distance), while `distance_meters` is simply
accumulated along whichever path Dijkstra selects on that basis. It performs
no database queries and knows nothing about coordinates, resources, or
response targets - NodeMappingService resolves coordinates to node ids
before this runs, and composing the final RouteResult (with resource_id/
response_target_id attached) happens after this runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

from src.models.graph_edge import GraphEdge
from src.models.routing import RouteStatus


@dataclass(frozen=True)
class DijkstraResult:
    """The outcome of one shortest-path search between two graph nodes."""

    status: RouteStatus
    node_path: tuple[int, ...]
    distance_meters: float | None
    travel_time_seconds: float | None


class DijkstraCalculator:
    """Computes the fastest (lowest total travel_time_seconds) path between two GraphNode ids."""

    def calculate_shortest_path(
        self,
        source_node_id: int,
        target_node_id: int,
        edges: Sequence[GraphEdge],
    ) -> DijkstraResult:
        """Return the fastest path from source_node_id to target_node_id.

        Runs Dijkstra's algorithm with each edge's travel_time_seconds as
        weight. Ties in cumulative travel time are broken by preferring the
        shorter cumulative distance_meters (less fuel, less exposure for the
        same ETA); node id is only the final fallback, so results stay
        deterministic even when two paths tie on both time and distance.
        If source_node_id == target_node_id, the resource is already at
        the target: returns a trivial zero-cost REACHABLE result without
        touching `edges`. Returns an UNREACHABLE result (empty node_path,
        distance/time both None) when no path connects the two nodes.
        """
        _validate_node_id("source_node_id", source_node_id)
        _validate_node_id("target_node_id", target_node_id)
        adjacency = _build_adjacency_list(edges)

        if source_node_id == target_node_id:
            return DijkstraResult(
                status=RouteStatus.REACHABLE,
                node_path=(source_node_id,),
                distance_meters=0.0,
                travel_time_seconds=0.0,
            )

        return _run_dijkstra(source_node_id, target_node_id, adjacency)


def _run_dijkstra(
    source_node_id: int,
    target_node_id: int,
    adjacency: dict[int, list[tuple[int, float, float]]],
) -> DijkstraResult:
    # Lexicographic cost per node: (cumulative_travel_time_seconds,
    # cumulative_distance_meters). Comparing/relaxing on this tuple means
    # travel time dominates, and distance only decides between paths that
    # tie exactly on time - never the other way around.
    best_cost: dict[int, tuple[float, float]] = {source_node_id: (0.0, 0.0)}
    predecessor: dict[int, int] = {}
    visited: set[int] = set()

    # (travel_time_seconds, distance_meters, node_id): node_id is a pure
    # last-resort tiebreaker so heap comparisons stay deterministic even
    # when two frontier entries tie on both time and distance.
    frontier: list[tuple[float, float, int]] = [(0.0, 0.0, source_node_id)]

    while frontier:
        current_time, current_distance, current_node = heapq.heappop(frontier)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == target_node_id:
            break

        for neighbor_id, edge_time, edge_distance in adjacency.get(current_node, []):
            if neighbor_id in visited:
                continue
            candidate_cost = (current_time + edge_time, current_distance + edge_distance)
            if candidate_cost < best_cost.get(neighbor_id, (math.inf, math.inf)):
                best_cost[neighbor_id] = candidate_cost
                predecessor[neighbor_id] = current_node
                heapq.heappush(frontier, (candidate_cost[0], candidate_cost[1], neighbor_id))

    if target_node_id not in visited:
        return DijkstraResult(
            status=RouteStatus.UNREACHABLE,
            node_path=(),
            distance_meters=None,
            travel_time_seconds=None,
        )

    target_time, target_distance = best_cost[target_node_id]
    return DijkstraResult(
        status=RouteStatus.REACHABLE,
        node_path=_reconstruct_path(predecessor, source_node_id, target_node_id),
        distance_meters=target_distance,
        travel_time_seconds=target_time,
    )


def _reconstruct_path(predecessor: dict[int, int], source_node_id: int, target_node_id: int) -> tuple[int, ...]:
    path = [target_node_id]
    while path[-1] != source_node_id:
        path.append(predecessor[path[-1]])
    path.reverse()
    return tuple(path)


def _build_adjacency_list(edges: Sequence[GraphEdge]) -> dict[int, list[tuple[int, float, float]]]:
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError(f"edges must be a sequence of GraphEdge, got {edges!r}")

    adjacency: dict[int, list[tuple[int, float, float]]] = {}
    for edge in edges:
        if not isinstance(edge, GraphEdge):
            raise ValueError(f"edges must contain only GraphEdge instances, got {edge!r}")
        _validate_non_negative_finite("travel_time_seconds", edge.travel_time_seconds, edge)
        _validate_non_negative_finite("distance_meters", edge.distance_meters, edge)
        adjacency.setdefault(edge.source_node_id, []).append(
            (edge.target_node_id, edge.travel_time_seconds, edge.distance_meters)
        )
    return adjacency


def _validate_node_id(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def _validate_non_negative_finite(field_name: str, value: object, edge: GraphEdge) -> None:
    # Dijkstra's correctness relies on non-negative edge weights - a
    # negative or non-finite travel_time_seconds/distance_meters would
    # silently corrupt the search rather than raise, so it is rejected here.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(
            f"edge {field_name} must be a non-negative finite number, got {value!r} "
            f"for edge {edge.source_node_id!r}->{edge.target_node_id!r}"
        )
