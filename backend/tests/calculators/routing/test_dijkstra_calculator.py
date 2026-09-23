"""Tests for pure deterministic shortest-path calculation."""
from __future__ import annotations

import pytest

from src.calculators.routing.dijkstra_calculator import DijkstraCalculator, DijkstraResult
from src.models.graph_edge import GraphEdge
from src.models.routing import RouteStatus


def make_edge(source_node_id: int, target_node_id: int, distance_meters: float, travel_time_seconds: float) -> GraphEdge:
    return GraphEdge(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        distance_meters=distance_meters,
        travel_time_seconds=travel_time_seconds,
    )


def test_reachable_path_over_a_simple_chain():
    edges = [
        make_edge(1, 2, distance_meters=900.0, travel_time_seconds=60.0),
        make_edge(2, 3, distance_meters=900.0, travel_time_seconds=60.0),
    ]

    result = DijkstraCalculator().calculate_shortest_path(1, 3, edges)

    assert result == DijkstraResult(
        status=RouteStatus.REACHABLE,
        node_path=(1, 2, 3),
        distance_meters=1800.0,
        travel_time_seconds=120.0,
    )


def test_unreachable_disconnected_graph():
    edges = [
        make_edge(1, 2, distance_meters=500.0, travel_time_seconds=30.0),
        # Separate component - no edge bridges {1, 2} and {3, 4}.
        make_edge(3, 4, distance_meters=500.0, travel_time_seconds=30.0),
    ]

    result = DijkstraCalculator().calculate_shortest_path(1, 4, edges)

    assert result == DijkstraResult(
        status=RouteStatus.UNREACHABLE,
        node_path=(),
        distance_meters=None,
        travel_time_seconds=None,
    )


def test_unreachable_when_source_has_no_outgoing_edges():
    edges = [make_edge(2, 1, distance_meters=500.0, travel_time_seconds=30.0)]

    result = DijkstraCalculator().calculate_shortest_path(1, 2, edges)

    assert result.status is RouteStatus.UNREACHABLE


def test_source_equal_to_target_is_trivially_reachable_without_edges():
    result = DijkstraCalculator().calculate_shortest_path(5, 5, edges=[])

    assert result == DijkstraResult(
        status=RouteStatus.REACHABLE,
        node_path=(5,),
        distance_meters=0.0,
        travel_time_seconds=0.0,
    )


def test_prefers_fastest_route_over_shorter_distance_and_fewer_hops():
    edges = [
        # One hop: short distance but very slow (e.g. unpaved/congested road).
        make_edge(1, 3, distance_meters=500.0, travel_time_seconds=1000.0),
        # Two hops: longer distance but much faster overall (e.g. highway).
        make_edge(1, 2, distance_meters=300.0, travel_time_seconds=100.0),
        make_edge(2, 3, distance_meters=300.0, travel_time_seconds=100.0),
    ]

    result = DijkstraCalculator().calculate_shortest_path(1, 3, edges)

    assert result.status is RouteStatus.REACHABLE
    assert result.node_path == (1, 2, 3)
    assert result.travel_time_seconds == 200.0
    assert result.distance_meters == 600.0


def test_handles_cycles_without_infinite_loop():
    edges = [
        make_edge(1, 2, distance_meters=100.0, travel_time_seconds=10.0),
        make_edge(2, 1, distance_meters=100.0, travel_time_seconds=10.0),
        make_edge(2, 3, distance_meters=100.0, travel_time_seconds=10.0),
    ]

    result = DijkstraCalculator().calculate_shortest_path(1, 3, edges)

    assert result.status is RouteStatus.REACHABLE
    assert result.node_path == (1, 2, 3)
    assert result.travel_time_seconds == 20.0
    assert result.distance_meters == 200.0


def test_tie_on_travel_time_is_broken_by_shorter_distance():
    # Two paths from 1 to 4 both take exactly 100s, but the path via node 3
    # is physically shorter than the path via node 2. Node 2 has the lower
    # id, so if the tiebreak were still node-id-first, node 2's path would
    # win - asserting on node 3's path proves distance is now checked
    # before node id.
    edges = [
        make_edge(1, 2, distance_meters=400.0, travel_time_seconds=50.0),
        make_edge(2, 4, distance_meters=400.0, travel_time_seconds=50.0),
        make_edge(1, 3, distance_meters=100.0, travel_time_seconds=50.0),
        make_edge(3, 4, distance_meters=100.0, travel_time_seconds=50.0),
    ]

    result = DijkstraCalculator().calculate_shortest_path(1, 4, edges)

    assert result.status is RouteStatus.REACHABLE
    assert result.travel_time_seconds == 100.0
    assert result.node_path == (1, 3, 4)
    assert result.distance_meters == 200.0


def test_tie_on_travel_time_and_distance_is_broken_deterministically_by_node_id():
    # Two paths from 1 to 4 tie on both travel time (100s) and distance
    # (2.0m) - the algorithm falls back to node id, deterministically
    # preferring the path via the lower-id node 2. Edge insertion order
    # deliberately favors 3 first, to prove the choice comes from the
    # algorithm's heap ordering, not from input order.
    edges = [
        make_edge(1, 3, distance_meters=1.0, travel_time_seconds=50.0),
        make_edge(3, 4, distance_meters=1.0, travel_time_seconds=50.0),
        make_edge(1, 2, distance_meters=1.0, travel_time_seconds=50.0),
        make_edge(2, 4, distance_meters=1.0, travel_time_seconds=50.0),
    ]

    result = DijkstraCalculator().calculate_shortest_path(1, 4, edges)

    assert result.status is RouteStatus.REACHABLE
    assert result.travel_time_seconds == 100.0
    assert result.node_path == (1, 2, 4)


def test_result_is_deterministic_across_repeated_calls():
    edges = [
        make_edge(1, 2, distance_meters=900.0, travel_time_seconds=60.0),
        make_edge(2, 3, distance_meters=900.0, travel_time_seconds=60.0),
        make_edge(1, 3, distance_meters=2500.0, travel_time_seconds=200.0),
    ]

    calculator = DijkstraCalculator()
    first_result = calculator.calculate_shortest_path(1, 3, edges)
    second_result = calculator.calculate_shortest_path(1, 3, edges)

    assert first_result == second_result


@pytest.mark.parametrize("node_id", [0, -1, True, "1", None, 1.5])
def test_invalid_source_node_id_rejected(node_id):
    with pytest.raises(ValueError):
        DijkstraCalculator().calculate_shortest_path(node_id, 2, edges=[])


@pytest.mark.parametrize("node_id", [0, -1, True, "1", None, 1.5])
def test_invalid_target_node_id_rejected(node_id):
    with pytest.raises(ValueError):
        DijkstraCalculator().calculate_shortest_path(1, node_id, edges=[])


def test_invalid_edges_type_rejected():
    with pytest.raises(ValueError):
        DijkstraCalculator().calculate_shortest_path(1, 2, edges="not-a-sequence")


def test_edges_containing_non_graph_edge_rejected():
    with pytest.raises(ValueError):
        DijkstraCalculator().calculate_shortest_path(1, 2, edges=[{"source_node_id": 1, "target_node_id": 2}])


def test_negative_travel_time_seconds_rejected():
    edges = [make_edge(1, 2, distance_meters=100.0, travel_time_seconds=-1.0)]

    with pytest.raises(ValueError):
        DijkstraCalculator().calculate_shortest_path(1, 2, edges)


def test_negative_distance_meters_rejected():
    edges = [make_edge(1, 2, distance_meters=-1.0, travel_time_seconds=10.0)]

    with pytest.raises(ValueError):
        DijkstraCalculator().calculate_shortest_path(1, 2, edges)


# ---------------------------------------------------------------------------
# RoutingGraph reuse (Optimization 1) - build_graph() + calculate_shortest_path_in_graph()
# ---------------------------------------------------------------------------


def test_build_graph_then_search_matches_calculate_shortest_path():
    edges = [
        make_edge(1, 2, distance_meters=900.0, travel_time_seconds=60.0),
        make_edge(2, 3, distance_meters=900.0, travel_time_seconds=60.0),
    ]
    calculator = DijkstraCalculator()
    graph = calculator.build_graph(edges)

    direct = calculator.calculate_shortest_path(1, 3, edges)
    via_graph = calculator.calculate_shortest_path_in_graph(1, 3, graph)

    assert direct == via_graph


def test_one_graph_supports_multiple_correct_searches():
    edges = [
        make_edge(1, 2, distance_meters=500.0, travel_time_seconds=60.0),
        make_edge(1, 3, distance_meters=5000.0, travel_time_seconds=600.0),
        make_edge(2, 1, distance_meters=500.0, travel_time_seconds=60.0),
    ]
    calculator = DijkstraCalculator()
    graph = calculator.build_graph(edges)

    result_1_to_2 = calculator.calculate_shortest_path_in_graph(1, 2, graph)
    result_1_to_3 = calculator.calculate_shortest_path_in_graph(1, 3, graph)
    result_2_to_1 = calculator.calculate_shortest_path_in_graph(2, 1, graph)

    assert result_1_to_2.status is RouteStatus.REACHABLE
    assert result_1_to_2.travel_time_seconds == 60.0
    assert result_1_to_3.status is RouteStatus.REACHABLE
    assert result_1_to_3.travel_time_seconds == 600.0
    assert result_2_to_1.status is RouteStatus.REACHABLE
    assert result_2_to_1.travel_time_seconds == 60.0


def test_directed_edges_are_not_reversed_via_graph_reuse():
    edges = [make_edge(1, 2, distance_meters=500.0, travel_time_seconds=30.0)]  # only 1 -> 2
    calculator = DijkstraCalculator()
    graph = calculator.build_graph(edges)

    forward = calculator.calculate_shortest_path_in_graph(1, 2, graph)
    backward = calculator.calculate_shortest_path_in_graph(2, 1, graph)

    assert forward.status is RouteStatus.REACHABLE
    assert backward.status is RouteStatus.UNREACHABLE


def test_source_equal_to_target_in_graph_is_trivially_reachable():
    calculator = DijkstraCalculator()
    graph = calculator.build_graph([])

    result = calculator.calculate_shortest_path_in_graph(5, 5, graph)

    assert result == DijkstraResult(
        status=RouteStatus.REACHABLE, node_path=(5,), distance_meters=0.0, travel_time_seconds=0.0
    )


def test_calculate_shortest_path_in_graph_rejects_non_routing_graph():
    calculator = DijkstraCalculator()
    with pytest.raises(ValueError):
        calculator.calculate_shortest_path_in_graph(1, 2, "not-a-graph")


def test_build_graph_validates_edges_same_as_calculate_shortest_path():
    calculator = DijkstraCalculator()
    with pytest.raises(ValueError):
        calculator.build_graph([{"source_node_id": 1, "target_node_id": 2}])


def test_reused_graph_result_is_deterministic_across_repeated_searches():
    edges = [
        make_edge(1, 2, distance_meters=900.0, travel_time_seconds=60.0),
        make_edge(2, 3, distance_meters=900.0, travel_time_seconds=60.0),
        make_edge(1, 3, distance_meters=2500.0, travel_time_seconds=200.0),
    ]
    calculator = DijkstraCalculator()
    graph = calculator.build_graph(edges)

    first = calculator.calculate_shortest_path_in_graph(1, 3, graph)
    second = calculator.calculate_shortest_path_in_graph(1, 3, graph)

    assert first == second
