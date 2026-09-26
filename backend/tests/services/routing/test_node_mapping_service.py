"""Tests for mapping resource/target coordinates onto road-network nodes."""
from __future__ import annotations

from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.response_target_type import ResponseTargetType
from src.models.routing import RoutingResource, RoutingTarget
from src.services.routing.node_mapping_service import NodeMappingService

# A tiny road network: node 1 -> node 2 -> node 3, laid out west to east.
NODE_1 = GraphNode(id=1, latitude=32.700, longitude=35.000)
NODE_2 = GraphNode(id=2, latitude=32.700, longitude=35.010)
NODE_3 = GraphNode(id=3, latitude=32.700, longitude=35.020)
EDGES = [
    GraphEdge(id=1, source_node_id=1, target_node_id=2, distance_meters=900.0, travel_time_seconds=60.0),
    GraphEdge(id=2, source_node_id=2, target_node_id=3, distance_meters=900.0, travel_time_seconds=60.0),
]


def make_resource(**overrides) -> RoutingResource:
    defaults = dict(resource_id="truck-1", station_id="station-1", latitude=32.700, longitude=35.000)
    defaults.update(overrides)
    return RoutingResource(**defaults)


def make_target(**overrides) -> RoutingTarget:
    defaults = dict(
        response_target_id=1,
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE.value,
        latitude=32.700,
        longitude=35.020,
        priority_score=100.0,
    )
    defaults.update(overrides)
    return RoutingTarget(**defaults)


def test_map_resource_to_nearest_node():
    service = NodeMappingService()
    resource = make_resource(latitude=32.700, longitude=35.0005)

    assert service.map_resource(resource, [NODE_1, NODE_2, NODE_3], EDGES) == 1


def test_map_target_to_nearest_node():
    service = NodeMappingService()
    target = make_target(latitude=32.700, longitude=35.0195)

    assert service.map_target(target, [NODE_1, NODE_2, NODE_3], EDGES) == 3


def test_map_resource_skips_node_with_no_outgoing_edge():
    # Node 3 is the endpoint of the network (only an incoming edge from node
    # 2) - a resource sitting right next to it must still map to node 2,
    # the nearest node it could actually depart from.
    service = NodeMappingService()
    resource = make_resource(latitude=32.700, longitude=35.0199)

    assert service.map_resource(resource, [NODE_1, NODE_2, NODE_3], EDGES) == 2


def test_map_target_skips_node_with_no_incoming_edge():
    # Node 1 is the start of the network (only an outgoing edge to node 2) -
    # a target sitting right next to it must still map to node 2, the
    # nearest node that anything could actually arrive at.
    service = NodeMappingService()
    target = make_target(latitude=32.700, longitude=35.0001)

    assert service.map_target(target, [NODE_1, NODE_2, NODE_3], EDGES) == 2


def test_map_resource_ignores_isolated_node_with_no_edges_at_all():
    isolated = GraphNode(id=99, latitude=32.700, longitude=35.000)
    service = NodeMappingService()
    resource = make_resource(latitude=32.700, longitude=35.000)

    assert service.map_resource(resource, [isolated, NODE_1, NODE_2, NODE_3], EDGES) == 1


def test_map_resource_returns_none_when_no_nodes():
    service = NodeMappingService()
    resource = make_resource()

    assert service.map_resource(resource, [], []) is None


def test_map_resource_returns_none_when_the_nearest_node_is_too_far_away():
    # The resource is genuinely ~90km from every node in this graph (the
    # fetched road network only covers a different area) - it must map to
    # None (UNMAPPABLE), never snap to a distant, unrelated node.
    service = NodeMappingService()
    resource = make_resource(latitude=31.900, longitude=35.000)

    assert service.map_resource(resource, [NODE_1, NODE_2, NODE_3], EDGES) is None


def test_map_target_returns_none_when_the_nearest_node_is_too_far_away():
    service = NodeMappingService()
    target = make_target(latitude=31.900, longitude=35.000)

    assert service.map_target(target, [NODE_1, NODE_2, NODE_3], EDGES) is None


def test_map_resource_maps_normally_just_inside_the_max_snap_distance():
    # ~4.9km south of node 1 - inside MAX_SNAP_DISTANCE_KM (5.0), so this
    # must still map normally, proving the cap does not trip prematurely.
    service = NodeMappingService()
    resource = make_resource(latitude=32.656, longitude=35.000)

    assert service.map_resource(resource, [NODE_1, NODE_2, NODE_3], EDGES) == 1


def test_map_target_returns_none_when_no_node_has_incoming_edge():
    # An edgeless graph: every node fails the "has an incoming edge" test.
    service = NodeMappingService()
    target = make_target()

    assert service.map_target(target, [NODE_1, NODE_2, NODE_3], []) is None


def test_map_resources_batches_multiple_resources():
    service = NodeMappingService()
    resources = [
        make_resource(resource_id="truck-1", latitude=32.700, longitude=35.0005),
        make_resource(resource_id="truck-2", latitude=32.700, longitude=35.0199),
    ]

    result = service.map_resources(resources, [NODE_1, NODE_2, NODE_3], EDGES)

    assert result == {"truck-1": 1, "truck-2": 2}


def test_map_targets_batches_multiple_targets():
    service = NodeMappingService()
    targets = [
        make_target(response_target_id=1, latitude=32.700, longitude=35.0195),
        make_target(response_target_id=2, latitude=32.700, longitude=35.0001),
    ]

    result = service.map_targets(targets, [NODE_1, NODE_2, NODE_3], EDGES)

    assert result == {1: 3, 2: 2}


def test_map_resource_logs_a_warning_when_the_snap_is_egregiously_far(caplog):
    # ~900m from node 1 - well past SNAP_DISTANCE_WARNING_KM (0.5km) but
    # still comfortably inside MAX_SNAP_DISTANCE_KM (5.0km), so the resource
    # still maps normally. The "Silent Failures" fix: this must never be
    # silent, since a genuinely local fetch's node should be within a few
    # hundred meters, not ~900m - see global_planning_input_builder.py's
    # _STATION_COVERAGE_CHECK_RADIUS_KM for the production case this guards.
    # Commit 0e3a3f8 deliberately lowered the guard from WARNING to DEBUG, so it
    # is captured at DEBUG here (the record must still be emitted).
    service = NodeMappingService()
    resource = make_resource(latitude=32.708, longitude=35.000)

    with caplog.at_level("DEBUG", logger="src.services.routing.node_mapping_service"):
        node_id = service.map_resource(resource, [NODE_1, NODE_2, NODE_3], EDGES)

    assert node_id == 1
    assert any("Snap-distance guard" in record.message for record in caplog.records)


def test_map_resource_does_not_warn_for_a_genuinely_close_snap(caplog):
    service = NodeMappingService()
    resource = make_resource(latitude=32.700, longitude=35.0005)  # ~46m from node 1

    # DEBUG, matching the guard's level (0e3a3f8) - otherwise this check would pass vacuously.
    with caplog.at_level("DEBUG", logger="src.services.routing.node_mapping_service"):
        node_id = service.map_resource(resource, [NODE_1, NODE_2, NODE_3], EDGES)

    assert node_id == 1
    assert not any("Snap-distance guard" in record.message for record in caplog.records)
