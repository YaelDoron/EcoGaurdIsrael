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

from sqlalchemy import event

from src.database.models.graph_node_db import GraphNodeDB
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.repositories.road_network_repository import RoadNetworkRepository


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


def test_save_network_handles_empty_nodes_and_edges(sqlite_session_factory):
    repository = RoadNetworkRepository()
    session = sqlite_session_factory()

    repository.save_network(session, [], [])

    stored_nodes, stored_edges = repository.get_network_in_bbox(session, -90.0, 90.0, -180.0, 180.0)
    session.close()

    assert stored_nodes == []
    assert stored_edges == []
