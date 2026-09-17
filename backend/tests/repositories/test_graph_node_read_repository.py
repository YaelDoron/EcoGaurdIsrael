"""Unit tests for GraphNodeReadRepository (Epic 6, US 6.3, Task 4) using SQLite in-memory."""
from __future__ import annotations

import pytest
from sqlalchemy import event

from src.database.models.graph_node_db import GraphNodeDB
from src.repositories.graph_node_read_repository import GraphNodeReadRepository


@pytest.fixture
def repository(sqlite_session_factory) -> GraphNodeReadRepository:
    return GraphNodeReadRepository(session_factory=sqlite_session_factory)


def seed_nodes(sqlite_session_factory, nodes: dict[int, tuple[float, float]]) -> None:
    session = sqlite_session_factory()
    try:
        for node_id, (latitude, longitude) in nodes.items():
            session.add(GraphNodeDB(id=node_id, latitude=latitude, longitude=longitude))
        session.commit()
    finally:
        session.close()


def test_returns_coordinates_for_existing_node_ids(repository, sqlite_session_factory):
    seed_nodes(sqlite_session_factory, {1: (32.0, 35.0), 2: (32.1, 35.1), 3: (32.2, 35.2)})

    nodes_by_id = repository.get_by_ids([1, 2, 3])

    assert nodes_by_id[1].latitude == 32.0
    assert nodes_by_id[1].longitude == 35.0
    assert nodes_by_id[2].latitude == 32.1
    assert nodes_by_id[3].longitude == 35.2


def test_missing_node_id_is_absent_from_result(repository, sqlite_session_factory):
    seed_nodes(sqlite_session_factory, {1: (32.0, 35.0)})

    nodes_by_id = repository.get_by_ids([1, 999])

    assert 1 in nodes_by_id
    assert 999 not in nodes_by_id


def test_empty_input_returns_empty_dict_without_querying(repository, sqlite_session_factory, sqlite_engine):
    query_count = 0

    def count_query(*args, **kwargs) -> None:
        nonlocal query_count
        query_count += 1

    event.listen(sqlite_engine, "before_cursor_execute", count_query)
    try:
        result = repository.get_by_ids([])
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", count_query)

    assert result == {}
    assert query_count == 0


def test_bulk_lookup_issues_a_single_query_not_one_per_node(repository, sqlite_session_factory, sqlite_engine):
    seed_nodes(sqlite_session_factory, {i: (float(i), float(i)) for i in range(1, 11)})

    select_statements = []

    def record_select(conn, cursor, statement, parameters, context, executemany) -> None:
        if statement.strip().upper().startswith("SELECT"):
            select_statements.append(statement)

    event.listen(sqlite_engine, "before_cursor_execute", record_select)
    try:
        nodes_by_id = repository.get_by_ids(range(1, 11))
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", record_select)

    assert len(nodes_by_id) == 10
    assert len(select_statements) == 1


def test_duplicate_node_ids_still_produce_one_query_and_correct_result(
    repository, sqlite_session_factory, sqlite_engine
):
    seed_nodes(sqlite_session_factory, {1: (32.0, 35.0), 2: (32.1, 35.1)})

    select_statements = []

    def record_select(conn, cursor, statement, parameters, context, executemany) -> None:
        if statement.strip().upper().startswith("SELECT"):
            select_statements.append(statement)

    event.listen(sqlite_engine, "before_cursor_execute", record_select)
    try:
        nodes_by_id = repository.get_by_ids([1, 1, 2, 2, 1])
    finally:
        event.remove(sqlite_engine, "before_cursor_execute", record_select)

    assert set(nodes_by_id.keys()) == {1, 2}
    assert len(select_statements) == 1
