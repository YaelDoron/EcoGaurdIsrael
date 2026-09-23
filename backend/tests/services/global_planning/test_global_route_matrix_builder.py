"""Tests for GlobalRouteMatrixBuilder (Stage 3 of the Global Multi-Incident
Optimizer refactor, Tasks 10-16, 22, 27, 30).

Critical acceptance test: a genuine CROSS-EVENT matrix (Task 16) - a
resource originally gathered near FireEvent A's local area must still get
a route computed to FireEvent B's target, not just its "own" event.
"""
from __future__ import annotations

import pytest

from src.calculators.routing.dijkstra_calculator import DijkstraCalculator
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.resource_status import ResourceStatus
from src.models.response_target_type import ResponseTargetType
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.routing.node_mapping_service import NodeMappingService
from src.utils.geo import haversine_distance_km

# Node layout: 1=station A origin, 2=station B origin, 3=target A1, 4=target B1;
# 6/7 form a disconnected component (isolated source -> isolated target) used
# to construct a genuinely UNREACHABLE (mappable but no path) pair.
NODE_STATION_A = 1
NODE_STATION_B = 2
NODE_TARGET_A1 = 3
NODE_TARGET_B1 = 4
NODE_ISOLATED_SOURCE = 6
NODE_ISOLATED_TARGET = 7


def _resource(resource_id, station_id, lat, lon, status=ResourceStatus.AVAILABLE) -> GlobalPlanningResource:
    return GlobalPlanningResource(
        resource_id=resource_id,
        station_id=station_id,
        station_name=station_id,
        station_latitude=lat,
        station_longitude=lon,
        operational_status=status,
    )


def _target(fire_event_id, response_target_id, lat, lon) -> GlobalPlanningTarget:
    return GlobalPlanningTarget(
        fire_event_id=fire_event_id,
        response_target_id=response_target_id,
        target_order=1,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=lat,
        longitude=lon,
        priority_score=100.0,
    )


def _nodes_and_edges(include_unreachable_target=False):
    nodes = [
        GraphNode(id=NODE_STATION_A, latitude=32.70, longitude=35.00),
        GraphNode(id=NODE_STATION_B, latitude=32.90, longitude=35.20),
        GraphNode(id=NODE_TARGET_A1, latitude=32.71, longitude=35.01),
        GraphNode(id=NODE_TARGET_B1, latitude=32.91, longitude=35.21),
    ]
    edges = [
        GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_A1, distance_meters=500.0, travel_time_seconds=60.0),
        GraphEdge(source_node_id=NODE_STATION_A, target_node_id=NODE_TARGET_B1, distance_meters=5000.0, travel_time_seconds=600.0),
        GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_A1, distance_meters=5000.0, travel_time_seconds=600.0),
        GraphEdge(source_node_id=NODE_STATION_B, target_node_id=NODE_TARGET_B1, distance_meters=500.0, travel_time_seconds=60.0),
    ]
    if include_unreachable_target:
        # A separate, disconnected component: node 7 DOES have an incoming
        # edge (from node 6), so it is a valid mapping target - but no edge
        # connects it to the {1,2,3,4} component at all, so Dijkstra from
        # any real station genuinely finds no path (UNREACHABLE, not just
        # unmapped).
        nodes.append(GraphNode(id=NODE_ISOLATED_SOURCE, latitude=33.5, longitude=36.0))
        nodes.append(GraphNode(id=NODE_ISOLATED_TARGET, latitude=33.51, longitude=36.01))
        edges.append(
            GraphEdge(
                source_node_id=NODE_ISOLATED_SOURCE,
                target_node_id=NODE_ISOLATED_TARGET,
                distance_meters=100.0,
                travel_time_seconds=10.0,
            )
        )
    return nodes, edges


def make_builder() -> GlobalRouteMatrixBuilder:
    return GlobalRouteMatrixBuilder(node_mapping_service=NodeMappingService(), dijkstra_calculator=DijkstraCalculator())


class _CountingDijkstraCalculator(DijkstraCalculator):
    """Spies on build_graph() calls (Optimization 1) so a test can assert
    the adjacency list is built exactly once per route-matrix build,
    regardless of how many (resource, target) pairs it searches."""

    def __init__(self):
        super().__init__()
        self.build_graph_calls = 0

    def build_graph(self, edges):
        self.build_graph_calls += 1
        return super().build_graph(edges)


# ---------------------------------------------------------------------------
# Task 16 - the critical cross-event matrix test
# ---------------------------------------------------------------------------


def test_cross_event_matrix_contains_all_four_resource_target_combinations():
    resource_a = _resource("R1", "STATION-A", 32.70, 35.00)
    resource_b = _resource("R2", "STATION-B", 32.90, 35.20)
    target_a1 = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    target_b1 = _target(fire_event_id=2, response_target_id=201, lat=32.91, lon=35.21)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource_a, resource_b), (target_a1, target_b1), nodes, edges)

    pairs = {(option.resource_id, option.response_target_id) for option in result.matrix}
    assert pairs == {("R1", 101), ("R1", 201), ("R2", 101), ("R2", 201)}
    # R1, gathered near event A, still has a real route to event B's target.
    cross_route = result.matrix.get("R1", 201)
    assert cross_route is not None
    assert cross_route.fire_event_id == 2
    assert cross_route.eta_seconds == 600.0


def test_cross_region_resource_still_routes_to_the_other_event():
    """Task 27: a resource geographically closer to A can still have a
    valid, correctly-attributed route to B."""
    resource_a = _resource("R1", "STATION-A", 32.70, 35.00)
    target_b1 = _target(fire_event_id=2, response_target_id=201, lat=32.91, lon=35.21)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource_a,), (target_b1,), nodes, edges)

    option = result.matrix.get("R1", 201)
    assert option is not None
    assert option.fire_event_id == 2
    assert option.route_distance_meters == 5000.0


# ---------------------------------------------------------------------------
# Same-station dedup (Task 14)
# ---------------------------------------------------------------------------


def test_resources_at_the_same_station_reuse_the_cached_dijkstra_result():
    resource_1 = _resource("R1", "STATION-A", 32.70, 35.00)
    resource_2 = _resource("R2", "STATION-A", 32.70, 35.00)  # same station/origin
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource_1, resource_2), (target,), nodes, edges)

    assert result.dijkstra_call_count == 1  # one distinct (source_node, target_node) pair
    assert result.matrix.get("R1", 101) is not None
    assert result.matrix.get("R2", 101) is not None
    assert result.matrix.get("R1", 101).eta_seconds == result.matrix.get("R2", 101).eta_seconds


# ---------------------------------------------------------------------------
# Task 15 - unreachable routes
# ---------------------------------------------------------------------------


def test_unreachable_pair_does_not_remove_other_feasible_pairs():
    resource = _resource("R1", "STATION-A", 32.70, 35.00)
    reachable_target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    unreachable_target = _target(fire_event_id=1, response_target_id=102, lat=33.51, lon=36.01)
    nodes, edges = _nodes_and_edges(include_unreachable_target=True)

    result = make_builder().build((resource,), (reachable_target, unreachable_target), nodes, edges)

    assert result.matrix.get("R1", 101) is not None
    assert result.matrix.get("R1", 102) is None
    assert result.potential_pairs == 2
    assert result.feasible_pairs == 1


def test_unmappable_resource_produces_no_route_without_failing_the_whole_build():
    """A road network with no edges at all -> every coordinate is
    UNMAPPABLE (no candidate node has an outgoing/incoming edge). The
    build must not raise; it simply produces zero routes."""
    resource = _resource("R1", "STATION-A", 32.70, 35.00)
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes = [GraphNode(id=NODE_STATION_A, latitude=32.70, longitude=35.00), GraphNode(id=NODE_TARGET_A1, latitude=32.71, longitude=35.01)]

    result = make_builder().build((resource,), (target,), nodes, [])

    assert len(result.matrix) == 0
    assert result.dijkstra_call_count == 0


# ---------------------------------------------------------------------------
# Task 4/22 - UNAVAILABLE exclusion, committed resources still routed
# ---------------------------------------------------------------------------


def test_unavailable_resource_gets_no_routes_computed():
    resource = _resource("R1", "STATION-A", 32.70, 35.00, status=ResourceStatus.UNAVAILABLE)
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource,), (target,), nodes, edges)

    assert result.resource_count == 0
    assert len(result.matrix) == 0


def test_committed_resource_still_gets_routes_to_both_events():
    committed_resource = GlobalPlanningResource(
        resource_id="R1", station_id="STATION-A", station_name="STATION-A",
        station_latitude=32.70, station_longitude=35.00, operational_status=ResourceStatus.ASSIGNED,
        current_commitment_fire_event_id=1, current_commitment_response_plan_id=10,
    )
    target_a1 = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    target_b1 = _target(fire_event_id=2, response_target_id=201, lat=32.91, lon=35.21)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((committed_resource,), (target_a1, target_b1), nodes, edges)

    assert result.matrix.get("R1", 101) is not None  # its own committed event
    assert result.matrix.get("R1", 201) is not None  # the OTHER event too - Stage 4 decides later


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_resources_produces_empty_matrix():
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((), (target,), nodes, edges)

    assert len(result.matrix) == 0
    assert result.dijkstra_call_count == 0


def test_empty_targets_produces_empty_matrix():
    resource = _resource("R1", "STATION-A", 32.70, 35.00)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource,), (), nodes, edges)

    assert len(result.matrix) == 0
    assert result.dijkstra_call_count == 0


# ---------------------------------------------------------------------------
# First-Mile Heuristic Fallback: a small, bounded correction added on TOP of
# an already-REACHABLE Dijkstra route when the origin had to snap to a real
# road node a non-trivial distance from the station's true coordinate (a
# station-micro fetch that exhausted every retry, in production). This is
# never a Haversine route SUBSTITUTE - test_input_builder_never_imports_the_
# haversine_route_fallback (in test_global_planning_input_builder.py) already
# guards that this module never imports HaversineFallbackCalculator.
# ---------------------------------------------------------------------------

# ~166m north of NODE_STATION_A (32.70, 35.00) - comfortably past
# FIRST_MILE_PENALTY_THRESHOLD_KM (100m) without being anywhere near
# NodeMappingService's own MAX_SNAP_DISTANCE_KM (5km) cap, so it still snaps
# to NODE_STATION_A normally.
FAR_STATION_LAT = 32.7015
FAR_STATION_LON = 35.00


def test_first_mile_penalty_added_when_the_origin_snaps_far_from_the_true_station_coordinate():
    resource = _resource("R1", "STATION-FAR", FAR_STATION_LAT, FAR_STATION_LON)
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource,), (target,), nodes, edges)

    option = result.matrix.get("R1", 101)
    assert option is not None
    gap_km = haversine_distance_km(FAR_STATION_LAT, FAR_STATION_LON, 32.70, 35.00)
    assert gap_km > 0.1  # the fixture must genuinely exceed the penalty threshold, or this test proves nothing
    expected_extra_seconds = (gap_km / 30.0) * 3600.0
    expected_extra_meters = gap_km * 1000.0
    # Base route (NODE_STATION_A -> NODE_TARGET_A1) is 500.0m / 60.0s - see _nodes_and_edges.
    assert option.eta_seconds == pytest.approx(60.0 + expected_extra_seconds)
    assert option.route_distance_meters == pytest.approx(500.0 + expected_extra_meters)


def test_no_first_mile_penalty_when_the_snap_gap_is_within_the_ordinary_threshold():
    # ~44m north of the node - well under FIRST_MILE_PENALTY_THRESHOLD_KM
    # (100m), the ordinary slack between a station's exact coordinate and
    # its nearest real road node - must not be penalized at all.
    resource = _resource("R1", "STATION-NEAR", 32.7004, 35.00)
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource,), (target,), nodes, edges)

    option = result.matrix.get("R1", 101)
    assert option is not None
    assert option.eta_seconds == 60.0
    assert option.route_distance_meters == 500.0


def test_first_mile_penalty_never_fabricates_a_route_for_an_unreachable_pair():
    """The penalty only ever corrects an already-REACHABLE route - it must
    never turn a genuinely UNREACHABLE pair into a feasible one."""
    resource = _resource("R1", "STATION-FAR", FAR_STATION_LAT, FAR_STATION_LON)
    unreachable_target = _target(fire_event_id=1, response_target_id=102, lat=33.51, lon=36.01)
    nodes, edges = _nodes_and_edges(include_unreachable_target=True)

    result = make_builder().build((resource,), (unreachable_target,), nodes, edges)

    assert result.matrix.get("R1", 102) is None
    assert len(result.matrix) == 0


def test_first_mile_penalty_is_identical_for_resources_sharing_the_same_far_snapped_station():
    resource_1 = _resource("R1", "STATION-FAR", FAR_STATION_LAT, FAR_STATION_LON)
    resource_2 = _resource("R2", "STATION-FAR", FAR_STATION_LAT, FAR_STATION_LON)
    target = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    nodes, edges = _nodes_and_edges()

    result = make_builder().build((resource_1, resource_2), (target,), nodes, edges)

    assert result.dijkstra_call_count == 1  # same-station dedup (Task 14) still holds
    option_1 = result.matrix.get("R1", 101)
    option_2 = result.matrix.get("R2", 101)
    assert option_1 is not None and option_2 is not None
    assert option_1.eta_seconds == option_2.eta_seconds
    assert option_1.eta_seconds > 60.0  # confirms the penalty was genuinely added, not silently dropped


def test_first_mile_penalty_helper_returns_zero_for_a_close_snap_or_a_missing_node():
    assert GlobalRouteMatrixBuilder._first_mile_penalty(32.70, 35.00, None) == (0.0, 0.0)
    close_node = GraphNode(id=1, latitude=32.70, longitude=35.00)
    assert GlobalRouteMatrixBuilder._first_mile_penalty(32.70, 35.00, close_node) == (0.0, 0.0)
# Optimization 1 (performance pass) - one adjacency build per matrix build
# ---------------------------------------------------------------------------


def test_adjacency_graph_is_built_exactly_once_per_route_matrix_build():
    counting_calculator = _CountingDijkstraCalculator()
    builder = GlobalRouteMatrixBuilder(
        node_mapping_service=NodeMappingService(), dijkstra_calculator=counting_calculator
    )
    resource_a = _resource("R1", "STATION-A", 32.70, 35.00)
    resource_b = _resource("R2", "STATION-B", 32.90, 35.20)
    target_a1 = _target(fire_event_id=1, response_target_id=101, lat=32.71, lon=35.01)
    target_b1 = _target(fire_event_id=2, response_target_id=201, lat=32.91, lon=35.21)
    nodes, edges = _nodes_and_edges()

    result = builder.build((resource_a, resource_b), (target_a1, target_b1), nodes, edges)

    assert counting_calculator.build_graph_calls == 1  # once for the whole matrix, not once per pair
    assert result.dijkstra_call_count == 4  # 2 resources x 2 targets, all distinct (source, target) pairs
