"""Tests for RoadNetworkRepository, focused on the Task A1.5 bulk-upsert fix.

Root cause this locks in: the previous implementation called
`Session.merge()` once per node/edge. Because every node carries a real
primary key (its OSM node id), `merge()` had to issue a synchronous
SELECT-by-primary-key round trip *per node* to decide insert vs. update -
for a real regional road network (tens of thousands of rows) against a
remote database, that dominated the multi-minute simulation stall this task
diagnoses. The statement-count test below is deliberately round-trip-count
based (not wall-clock-based, which would be flaky) - it fails under the old
per-object merge() loop and passes under the new batched upsert.
"""
from __future__ import annotations

import math

from sqlalchemy import event

from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.repositories import road_network_repository as road_network_repository_module
from src.repositories.road_network_repository import EDGE_QUERY_CHUNK_SIZE, RoadNetworkRepository


def make_nodes(count: int, *, id_offset: int = 0, latitude: float = 32.7) -> list[GraphNode]:
    return [
        GraphNode(id=id_offset + index, latitude=latitude, longitude=35.0 + index * 0.001)
        for index in range(count)
    ]


def make_edges(node_ids: list[int]) -> list[GraphEdge]:
    return [
        GraphEdge(
            source_node_id=node_ids[index],
            target_node_id=node_ids[index + 1],
            distance_meters=100.0,
            travel_time_seconds=10.0,
        )
        for index in range(len(node_ids) - 1)
    ]


def test_save_network_persists_new_nodes_and_edges(sqlite_session_factory):
    repository = RoadNetworkRepository()
    nodes = make_nodes(5, id_offset=1000)
    edges = make_edges([node.id for node in nodes])

    session = sqlite_session_factory()
    repository.save_network(session, nodes, edges)

    stored_nodes, stored_edges = repository.get_network_in_bbox(session, 32.0, 33.0, 34.0, 36.0)
    session.close()

    assert {node.id for node in stored_nodes} == {node.id for node in nodes}
    assert len(stored_edges) == len(edges)


def test_save_network_upserts_existing_nodes_instead_of_raising(sqlite_session_factory):
    """Re-importing an overlapping bbox must update the existing node's
    coordinates in place, not raise a primary-key violation (the original
    merge()-based behavior this preserves)."""
    repository = RoadNetworkRepository()
    original = [GraphNode(id=42, latitude=32.7, longitude=35.0)]

    session = sqlite_session_factory()
    repository.save_network(session, original, [])

    moved = [GraphNode(id=42, latitude=33.5, longitude=36.5)]
    repository.save_network(session, moved, [])

    stored_nodes, _ = repository.get_network_in_bbox(session, 33.0, 34.0, 36.0, 37.0)
    session.close()

    assert len(stored_nodes) == 1
    assert stored_nodes[0].id == 42
    assert stored_nodes[0].latitude == 33.5
    assert stored_nodes[0].longitude == 36.5


def test_save_network_does_not_issue_one_round_trip_per_row(sqlite_session_factory, sqlite_engine):
    """The defect this test guards against: a per-object Session.merge()
    loop issues roughly one SELECT/INSERT statement per node/edge. For 60
    nodes + 59 edges that would be on the order of 60+ statements; the
    batched upsert/insert should need only a small constant number
    regardless of row count."""
    statement_count = 0

    @event.listens_for(sqlite_engine, "before_cursor_execute")
    def _count_statements(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        nonlocal statement_count
        statement_count += 1

    repository = RoadNetworkRepository()
    nodes = make_nodes(60, id_offset=5000)
    edges = make_edges([node.id for node in nodes])

    session = sqlite_session_factory()
    repository.save_network(session, nodes, edges)
    session.close()

    assert statement_count <= 10, (
        f"save_network executed {statement_count} statements for {len(nodes)} nodes + {len(edges)} edges; "
        "expected a small constant number of batched statements, not one per row."
    )


def test_save_network_and_lookup_are_correct_across_a_batch_boundary(sqlite_session_factory, monkeypatch):
    """Regression guard for the PostgreSQL 65535-bind-parameter crash
    (observed live on a real 4-concurrent-event bbox once accumulated OSM
    coverage grew large): _upsert_nodes and _edges_touching now chunk their
    queries at _BULK_OPERATION_BATCH_SIZE. Forces a tiny batch size so a
    modest node/edge set spans multiple batches, and asserts every node and
    edge - including one whose source is in one batch and target in
    another - still round-trips correctly, not just that saving succeeds."""
    monkeypatch.setattr(road_network_repository_module, "_BULK_OPERATION_BATCH_SIZE", 3)

    repository = RoadNetworkRepository()
    nodes = make_nodes(10, id_offset=9000)
    node_ids = [node.id for node in nodes]
    edges = make_edges(node_ids)  # chains node i -> node i+1, so edges straddle every batch boundary.

    session = sqlite_session_factory()
    repository.save_network(session, nodes, edges)

    stored_nodes, stored_edges = repository.get_network_in_bbox(session, 32.0, 33.0, 34.0, 36.0)
    session.close()

    assert {node.id for node in stored_nodes} == set(node_ids)
    assert len(stored_edges) == len(edges)
    stored_pairs = {(edge.source_node_id, edge.target_node_id) for edge in stored_edges}
    expected_pairs = {(edge.source_node_id, edge.target_node_id) for edge in edges}
    assert stored_pairs == expected_pairs


def test_save_network_handles_empty_nodes_and_edges(sqlite_session_factory):
    repository = RoadNetworkRepository()
    session = sqlite_session_factory()

    repository.save_network(session, [], [])

    stored_nodes, stored_edges = repository.get_network_in_bbox(session, -90.0, 90.0, -180.0, 180.0)
    session.close()

    assert stored_nodes == []
    assert stored_edges == []


# --- get_network_in_bbox() edge-filtering regression tests ---------------
#
# Root cause this section guards against: a bbox spanning multiple active
# incidents over the (intentionally never-purged) road-network cache can
# select tens of thousands of nodes. The previous implementation queried
# `GraphEdgeDB.source_node_id.in_(node_ids)` AND `.target_node_id.in_(node_ids)`
# in a single statement, binding ~2x the node count as parameters - which
# exceeded PostgreSQL's ~65,535 per-statement parameter limit at ~38,000
# nodes. The fix chunks only the source-node `.in_()` query and filters the
# target endpoint in Python against a `set`.

BBOX = (32.0, 33.0, 34.0, 36.0)  # min_lat, max_lat, min_lon, max_lon


def test_get_network_in_bbox_returns_empty_when_no_nodes_match(sqlite_session_factory):
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()

    nodes, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert nodes == []
    assert edges == []


def test_get_network_in_bbox_single_node_no_edges(sqlite_session_factory):
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    repository.save_network(session, [GraphNode(id=1, latitude=32.5, longitude=35.0)], [])

    nodes, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert {node.id for node in nodes} == {1}
    assert edges == []


def test_edge_with_both_endpoints_inside_bbox_is_included(sqlite_session_factory):
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    inside_nodes = [
        GraphNode(id=1, latitude=32.1, longitude=34.1),
        GraphNode(id=2, latitude=32.9, longitude=35.9),
    ]
    edge = GraphEdge(source_node_id=1, target_node_id=2, distance_meters=100.0, travel_time_seconds=10.0)
    repository.save_network(session, inside_nodes, [edge])

    _, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert len(edges) == 1
    assert edges[0].source_node_id == 1
    assert edges[0].target_node_id == 2


def test_edge_with_source_inside_target_outside_bbox_is_excluded(sqlite_session_factory):
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    nodes = [
        GraphNode(id=1, latitude=32.5, longitude=35.0),  # inside
        GraphNode(id=2, latitude=50.0, longitude=35.0),  # outside
    ]
    edge = GraphEdge(source_node_id=1, target_node_id=2, distance_meters=100.0, travel_time_seconds=10.0)
    repository.save_network(session, nodes, [edge])

    nodes_out, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert {node.id for node in nodes_out} == {1}
    assert edges == []


def test_edge_with_source_outside_target_inside_bbox_is_excluded(sqlite_session_factory):
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    nodes = [
        GraphNode(id=1, latitude=50.0, longitude=35.0),  # outside
        GraphNode(id=2, latitude=32.5, longitude=35.0),  # inside
    ]
    edge = GraphEdge(source_node_id=1, target_node_id=2, distance_meters=100.0, travel_time_seconds=10.0)
    repository.save_network(session, nodes, [edge])

    nodes_out, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert {node.id for node in nodes_out} == {2}
    assert edges == []


def test_directed_edge_direction_and_weights_are_preserved(sqlite_session_factory):
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    nodes = [
        GraphNode(id=1, latitude=32.1, longitude=34.1),
        GraphNode(id=2, latitude=32.9, longitude=35.9),
    ]
    # Only one direction is persisted - the reverse must not appear.
    edge = GraphEdge(source_node_id=1, target_node_id=2, distance_meters=123.5, travel_time_seconds=45.6)
    repository.save_network(session, nodes, [edge])

    _, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert len(edges) == 1
    assert edges[0].source_node_id == 1
    assert edges[0].target_node_id == 2
    assert edges[0].distance_meters == 123.5
    assert edges[0].travel_time_seconds == 45.6
    assert not any(e.source_node_id == 2 and e.target_node_id == 1 for e in edges)


def test_get_edges_within_node_set_deduplicates_when_forced_into_singleton_chunks(sqlite_session_factory, monkeypatch):
    """With EDGE_QUERY_CHUNK_SIZE forced to 1, every node is queried in its
    own chunk. An edge must still be returned exactly once, never once per
    chunk that could plausibly have matched it."""
    monkeypatch.setattr(road_network_repository_module, "EDGE_QUERY_CHUNK_SIZE", 1)
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    nodes = [
        GraphNode(id=1, latitude=32.1, longitude=34.1),
        GraphNode(id=2, latitude=32.9, longitude=35.9),
    ]
    edge = GraphEdge(source_node_id=1, target_node_id=2, distance_meters=100.0, travel_time_seconds=10.0)
    repository.save_network(session, nodes, [edge])

    _, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert len(edges) == 1


def test_multiple_chunks_preserve_all_expected_edges(sqlite_session_factory, monkeypatch):
    """Force a small chunk size across a node set larger than one chunk and
    confirm every valid edge is still returned, none lost, none duplicated."""
    monkeypatch.setattr(road_network_repository_module, "EDGE_QUERY_CHUNK_SIZE", 2)
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()

    nodes = [GraphNode(id=index, latitude=32.5, longitude=35.0 + index * 0.001) for index in range(1, 8)]
    edges = [
        GraphEdge(source_node_id=source, target_node_id=target, distance_meters=10.0, travel_time_seconds=1.0)
        for source, target in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 7)]
    ]
    repository.save_network(session, nodes, edges)

    _, stored_edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    expected = {(source, target) for source, target in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 7)]}
    assert {(e.source_node_id, e.target_node_id) for e in stored_edges} == expected


def test_cross_chunk_edge_is_still_included(sqlite_session_factory, monkeypatch):
    """Source and target forced into different (singleton) chunks - the
    target check must use the full node-id set, not the source's own chunk,
    or this edge would be silently dropped."""
    monkeypatch.setattr(road_network_repository_module, "EDGE_QUERY_CHUNK_SIZE", 1)
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()
    nodes = [
        GraphNode(id=10, latitude=32.1, longitude=34.1),
        GraphNode(id=20, latitude=32.9, longitude=35.9),
    ]
    edge = GraphEdge(source_node_id=10, target_node_id=20, distance_meters=50.0, travel_time_seconds=5.0)
    repository.save_network(session, nodes, [edge])

    _, edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    assert [(e.source_node_id, e.target_node_id) for e in edges] == [(10, 20)]


def test_large_node_set_never_exceeds_configured_parameter_count(sqlite_session_factory, sqlite_engine):
    """The core regression test: a node set far larger than PostgreSQL's
    practical single-query safety range must not produce one query with
    ~2 * node_count bound parameters. Uses a large synthetic id set with an
    empty edges table - no need to materialize tens of thousands of real
    rows to prove the query shape is safe."""
    session = sqlite_session_factory()
    repository = RoadNetworkRepository()

    large_node_count = 45_000
    node_ids = set(range(1, large_node_count + 1))
    assert large_node_count > EDGE_QUERY_CHUNK_SIZE, "test requires a node set spanning multiple chunks"

    captured_parameter_counts: list[int] = []

    @event.listens_for(sqlite_engine, "before_cursor_execute")
    def _record_parameter_count(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        if "graph_edges" in statement and "SELECT" in statement.upper():
            captured_parameter_counts.append(len(parameters))

    edges = repository._get_edges_within_node_set(session, node_ids)
    session.close()

    assert edges == []
    assert captured_parameter_counts, "expected at least one SELECT against graph_edges"
    assert all(count <= EDGE_QUERY_CHUNK_SIZE for count in captured_parameter_counts), (
        f"a chunked query bound {max(captured_parameter_counts)} parameters, "
        f"exceeding EDGE_QUERY_CHUNK_SIZE={EDGE_QUERY_CHUNK_SIZE}; the old implementation "
        "would have bound ~2x the full node count in one statement here."
    )
    expected_chunk_count = math.ceil(large_node_count / EDGE_QUERY_CHUNK_SIZE)
    assert len(captured_parameter_counts) == expected_chunk_count


def test_new_implementation_matches_naive_dual_in_query_for_small_data(sqlite_session_factory):
    """Cross-check against the pre-fix query shape (safe to run directly for
    small data) to confirm the chunked/filtered implementation is
    semantically identical, not just non-crashing."""
    from sqlalchemy import select

    session = sqlite_session_factory()
    repository = RoadNetworkRepository()

    nodes = [GraphNode(id=index, latitude=32.5, longitude=35.0 + index * 0.001) for index in range(1, 6)]
    outside_node = GraphNode(id=99, latitude=50.0, longitude=35.0)
    edges = [
        GraphEdge(source_node_id=1, target_node_id=2, distance_meters=1.0, travel_time_seconds=1.0),
        GraphEdge(source_node_id=2, target_node_id=3, distance_meters=1.0, travel_time_seconds=1.0),
        GraphEdge(source_node_id=3, target_node_id=99, distance_meters=1.0, travel_time_seconds=1.0),  # target outside
        GraphEdge(source_node_id=99, target_node_id=4, distance_meters=1.0, travel_time_seconds=1.0),  # source outside
    ]
    repository.save_network(session, nodes + [outside_node], edges)

    db_nodes = session.execute(
        select(GraphNodeDB).where(
            GraphNodeDB.latitude > BBOX[0],
            GraphNodeDB.latitude < BBOX[1],
            GraphNodeDB.longitude > BBOX[2],
            GraphNodeDB.longitude < BBOX[3],
        )
    ).scalars().all()
    node_ids = {n.id for n in db_nodes}
    naive_edges = session.execute(
        select(GraphEdgeDB).where(
            GraphEdgeDB.source_node_id.in_(node_ids),
            GraphEdgeDB.target_node_id.in_(node_ids),
        )
    ).scalars().all()
    naive_pairs = {(e.source_node_id, e.target_node_id) for e in naive_edges}

    _, new_edges = repository.get_network_in_bbox(session, *BBOX)
    session.close()

    new_pairs = {(e.source_node_id, e.target_node_id) for e in new_edges}
    assert new_pairs == naive_pairs
    assert new_pairs == {(1, 2), (2, 3)}
