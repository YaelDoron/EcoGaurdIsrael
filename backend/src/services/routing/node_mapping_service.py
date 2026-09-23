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

A coordinate also maps to None when even its NEAREST usable candidate is
farther than `MAX_SNAP_DISTANCE_KM` away (see that constant). Without this
cap, a coordinate whose real position falls outside the fetched road-network
graph's coverage (e.g. a station far from the fire, when the graph was only
fetched for the fire's immediate vicinity) would still snap to whatever
node happens to be geographically nearest among an unrelated, possibly
sparse local cluster - producing a route that is technically "mapped" but
bears no real relationship to the resource's/target's actual location
(rendering as a geometrically nonsensical path, and occasionally two
genuinely distant real-world points snapping to the very same node, which
then reports a misleading 0m/0s - see haversine_fallback_calculator.py for
that half of the fix). Mapping to None here is honest: "no node in the
fetched graph is close enough to represent this coordinate" is a real
UNMAPPABLE condition, not a snapping failure to paper over.
"""
from __future__ import annotations

import logging

from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.routing import RoutingResource, RoutingTarget
from src.utils.geo import haversine_distance_km

logger = logging.getLogger(__name__)

# Generous on purpose: a coordinate whose road-network graph was fetched to
# actually cover it should have a matching node within a few hundred meters
# to a couple of km, not this. This only trips for the genuine
# out-of-coverage case described above. Matches
# OperationalContextService.DEFAULT_OPERATIONAL_RADIUS_KM's own notion of
# "practically close" for the same kind of station-proximity judgment.
MAX_SNAP_DISTANCE_KM = 5.0

# Distance guard (the "Silent Failures" fix): a snap under MAX_SNAP_DISTANCE_KM
# still succeeds and produces a GlobalRouteOption, but a coordinate snapping
# to a node this far away is exactly the "route starts 1-2km from the
# station's blue dot" symptom - a real, working route, just quietly built
# from the wrong origin/destination node because nothing genuinely local was
# in the fetched graph. Below MAX_SNAP_DISTANCE_KM this never raises or
# blocks the route; it only makes an otherwise-silent quality problem show
# up in logs so a missing/failed targeted fetch (station-micro, per-anchor,
# etc.) is diagnosable instead of only visible as "the map looks a bit off."
SNAP_DISTANCE_WARNING_KM = 0.5


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
        return self._nearest_node_id(
            resource.latitude, resource.longitude, candidates, entity_label=f"resource station {resource.station_id}"
        )

    def map_target(
        self,
        target: RoutingTarget,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> int | None:
        """Return the id of the nearest GraphNode with an incoming edge, or None if none exists."""
        candidates = self._nodes_with_incoming_edges(nodes, edges)
        return self._nearest_node_id(
            target.latitude, target.longitude, candidates, entity_label=f"target {target.response_target_id}"
        )

    def map_resources(
        self,
        resources: list[RoutingResource],
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> dict[str, int | None]:
        """Return {resource_id: nearest_node_id_or_None}, reusing one outgoing-edge index for all resources."""
        candidates = self._nodes_with_outgoing_edges(nodes, edges)
        return {
            resource.resource_id: self._nearest_node_id(
                resource.latitude,
                resource.longitude,
                candidates,
                entity_label=f"resource station {resource.station_id}",
            )
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
            target.response_target_id: self._nearest_node_id(
                target.latitude,
                target.longitude,
                candidates,
                entity_label=f"target {target.response_target_id}",
            )
            for target in targets
        }

    @staticmethod
    def _nearest_node_id(
        latitude: float, longitude: float, candidates: list[GraphNode], *, entity_label: str
    ) -> int | None:
        if not candidates:
            return None
        nearest = min(
            candidates,
            key=lambda node: haversine_distance_km(latitude, longitude, node.latitude, node.longitude),
        )
        snap_distance_km = haversine_distance_km(latitude, longitude, nearest.latitude, nearest.longitude)
        if snap_distance_km > MAX_SNAP_DISTANCE_KM:
            return None
        if snap_distance_km > SNAP_DISTANCE_WARNING_KM:
            logger.warning(
                "Snap-distance guard: %s at (%.6f, %.6f) snapped to node %s %.0fm away "
                "(warning threshold %.0fm) - the road network near its actual coordinate "
                "may be missing from the fetched graph; the resulting route will start/end "
                "from this node, not the real location.",
                entity_label,
                latitude,
                longitude,
                nearest.id,
                snap_distance_km * 1000.0,
                SNAP_DISTANCE_WARNING_KM * 1000.0,
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
