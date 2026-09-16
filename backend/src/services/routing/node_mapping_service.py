"""Map resource/target coordinates to the nearest reachable road-network node.

NodeMappingService only answers "which GraphNode should stand in for this
coordinate for routing purposes?". It does not compute distances or paths
between nodes (that is the later pathfinding step, e.g. Dijkstra) and does
not decide REACHABLE vs UNREACHABLE (that only happens once a route between
two already-mapped nodes has actually been searched for). A coordinate maps
to None when the graph has no node that could serve as a route endpoint for
it - callers use that to build an UNMAPPABLE RouteResult.

"Reachable" here means direction-aware: a RoutingResource (a route's start)
must map to a node with at least one outgoing edge, and a RoutingTarget (a
route's end) must map to a node with at least one incoming edge - mapping to
a node with no edge in the needed direction would guarantee no path could
ever be found from/to it, regardless of how close it is.
"""
from __future__ import annotations

from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.routing import RoutingResource, RoutingTarget
from src.utils.geo import haversine_distance_km


class NodeMappingService:
    """Maps RoutingResource/RoutingTarget coordinates onto road-network GraphNodes."""

    def map_resource(
        self,
        resource: RoutingResource,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> int | None:
        """Return the id of the nearest GraphNode with an outgoing edge, or None if none exists."""
        candidates = self._nodes_with_outgoing_edges(nodes, edges)
        return self._nearest_node_id(resource.latitude, resource.longitude, candidates)

    def map_target(
        self,
        target: RoutingTarget,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> int | None:
        """Return the id of the nearest GraphNode with an incoming edge, or None if none exists."""
        candidates = self._nodes_with_incoming_edges(nodes, edges)
        return self._nearest_node_id(target.latitude, target.longitude, candidates)

    def map_resources(
        self,
        resources: list[RoutingResource],
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> dict[str, int | None]:
        """Return {resource_id: nearest_node_id_or_None}, reusing one outgoing-edge index for all resources."""
        candidates = self._nodes_with_outgoing_edges(nodes, edges)
        return {
            resource.resource_id: self._nearest_node_id(resource.latitude, resource.longitude, candidates)
            for resource in resources
        }

    def map_targets(
        self,
        targets: list[RoutingTarget],
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> dict[int, int | None]:
        """Return {response_target_id: nearest_node_id_or_None}, reusing one incoming-edge index for all targets."""
        candidates = self._nodes_with_incoming_edges(nodes, edges)
        return {
            target.response_target_id: self._nearest_node_id(target.latitude, target.longitude, candidates)
            for target in targets
        }

    @staticmethod
    def _nearest_node_id(latitude: float, longitude: float, candidates: list[GraphNode]) -> int | None:
        if not candidates:
            return None
        nearest = min(
            candidates,
            key=lambda node: haversine_distance_km(latitude, longitude, node.latitude, node.longitude),
        )
        return nearest.id

    @staticmethod
    def _nodes_with_outgoing_edges(nodes: list[GraphNode], edges: list[GraphEdge]) -> list[GraphNode]:
        source_node_ids = {edge.source_node_id for edge in edges}
        return [node for node in nodes if node.id in source_node_ids]

    @staticmethod
    def _nodes_with_incoming_edges(nodes: list[GraphNode], edges: list[GraphEdge]) -> list[GraphNode]:
        target_node_ids = {edge.target_node_id for edge in edges}
        return [node for node in nodes if node.id in target_node_ids]
