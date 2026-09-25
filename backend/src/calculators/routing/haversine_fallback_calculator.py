"""Straight-line (Haversine) fallback distance/ETA for a degenerate routing result.

`DijkstraCalculator` knows only graph node ids, never real-world
coordinates - `NodeMappingService` maps each resource/target coordinate to
its NEAREST available graph node, with no maximum-distance check. When the
fetched road-network graph for a station/event pair is sparse relative to
the real distance between them (e.g. the station sits far outside the
graph's coverage), two genuinely distant real-world points can both snap to
the SAME single nearest node. Dijkstra's own same-node shortcut then
correctly reports a graph distance/time of exactly 0 for that node pair -
which is a real graph-level answer, but a misleading one once the *actual*
coordinates are compared and turn out not to be the same point at all.

This module is the deliberate, explicit off-road estimate used ONLY for
that degenerate case:
- It is never used for a genuinely reachable node-to-node path with a
  nonzero result (that already has a real, routed distance/time - this
  module is not consulted at all).
- It is never used for UNMAPPABLE - an endpoint with no usable graph node
  keeps its existing `None`/`None` semantics (there is no node to build a
  path from).
- It IS used for UNREACHABLE (both endpoints mapped to a node, but the
  fetched graph is disconnected between them - Dijkstra found no path), so
  a disconnected graph no longer leaves a fire with no response actions.
  That case uses DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH and produces a
  straight line between the two mapped nodes, never a routed path, and only
  within MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM - see
  `estimate_disconnected_graph_distance_and_eta`.
- It is never used when the two coordinates truly are (near-)identical -
  see `MIN_DISTINCT_DISTANCE_METERS` - a real 0m/0s is not "wrong" then.
"""
from __future__ import annotations

from src.utils.geo import haversine_distance_km

# A rough, deliberately conservative average speed for a fire truck driving
# cross-country/off-road (no known road connects the two points in the
# fetched graph) - not a paved-road speed. Centralized so it can be
# recalibrated without touching callers.
DEFAULT_OFF_ROAD_SPEED_KMH = 40.0

# Speed for a disconnected-graph (UNREACHABLE) fallback. Deliberately slower
# than DEFAULT_OFF_ROAD_SPEED_KMH: there, a road-network answer of 0m/0s
# merely needs correcting, while here NO road connects the two points in
# the fetched graph at all, so a straight line (which is always shorter than
# any real route) needs a more conservative speed to avoid understating the ETA.
DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH = 30.0

# A straight line is only a plausible stand-in for a route between NEARBY
# points. Beyond this air distance a disconnected pair stays unreachable
# (no fallback) instead of being sent a resource across terrain no road
# connects at all - otherwise a station in a different region becomes a
# "feasible" candidate for a far-away fire purely because the fetched graph
# happens to be split. Matches the widest candidate search radius the global
# planner itself ever considers (GlobalCandidateResourceConfig.search_radii_km
# = 5/20/50 km), so the fallback never reaches further than candidate
# collection already does.
MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM = 50.0

# Below this real-world separation, a 0m/0s routing result is not
# degenerate - the resource and target genuinely are (practically) at the
# same point, so no fallback estimate is substituted.
MIN_DISTINCT_DISTANCE_METERS = 10.0


def is_degenerate_zero_result(distance_meters: float, travel_time_seconds: float) -> bool:
    """True when a REACHABLE Dijkstra result is exactly zero in either field.

    A real routed distance/time is only ever exactly 0 via
    `DijkstraCalculator`'s own source_node_id == target_node_id shortcut -
    genuine road-network paths always accumulate a positive distance/time
    over at least one edge. Any other reachable result is left untouched.
    """
    return distance_meters == 0.0 or travel_time_seconds == 0.0


def estimate_off_road_distance_and_eta(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    *,
    speed_kmh: float = DEFAULT_OFF_ROAD_SPEED_KMH,
) -> tuple[float, float]:
    """Return (distance_meters, travel_time_seconds) as a straight-line
    estimate at a flat off-road speed.

    A deliberately rough, last-resort estimate for the degenerate case
    described in this module's docstring - never a substitute for a real
    routed path, and never applied by this function itself (callers decide
    when the fallback applies via `is_degenerate_zero_result` and their own
    distinctness check).
    """
    distance_km = haversine_distance_km(
        origin_latitude, origin_longitude, destination_latitude, destination_longitude
    )
    distance_meters = distance_km * 1000.0
    travel_time_seconds = (distance_km / speed_kmh) * 3600.0
    return distance_meters, travel_time_seconds


def estimate_disconnected_graph_distance_and_eta(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
) -> tuple[float, float] | None:
    """Return (distance_meters, travel_time_seconds) for a resource/target
    pair Dijkstra found NO road path between (UNREACHABLE) - the straight-
    line distance between the two real coordinates at
    DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH - or None when the two points are
    farther apart than MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM (the
    pair then stays unreachable).

    The station's and target's own coordinates are used (not their snapped
    graph nodes), so the estimate also covers any gap between a coordinate
    and its nearest node - callers must therefore NOT add a first-mile
    penalty on top of it.
    """
    air_distance_km = haversine_distance_km(
        origin_latitude, origin_longitude, destination_latitude, destination_longitude
    )
    if air_distance_km > MAX_DISCONNECTED_GRAPH_FALLBACK_DISTANCE_KM:
        return None
    return estimate_off_road_distance_and_eta(
        origin_latitude,
        origin_longitude,
        destination_latitude,
        destination_longitude,
        speed_kmh=DISCONNECTED_GRAPH_FALLBACK_SPEED_KMH,
    )
