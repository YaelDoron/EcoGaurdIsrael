"""Persistence layer for the road network graph (nodes and edges).

RoadNetworkRepository bridges the Pydantic domain models (GraphNode,
GraphEdge) and the SQLAlchemy ORM models (GraphNodeDB, GraphEdgeDB). It is
responsible only for persistence access: no OSM parsing, graph-building, or
routing/pathfinding logic lives here.

Unlike the other repositories in this codebase, this one takes an explicit
`db: Session` argument per call rather than owning its own session factory.
This matches a typical FastAPI request-scoped session (e.g. a `Depends`
dependency handing out one `Session` per request) - the caller owns the
session and transaction lifecycle, this repository only issues queries and
writes against it.
"""
from __future__ import annotations

from sqlalchemy import insert, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode

# PostgreSQL hard-limits a single prepared statement to 65535 bind
# parameters. get_network_in_bbox's edge lookup used to pass one parameter
# per node id per IN(...) clause, and _upsert_nodes' multi-row VALUES
# INSERT passes 3 parameters per node in ONE statement; with several
# concurrent, geographically spread FireEvents (Global Multi-Incident
# Optimizer) plus accumulated per-region OSM coverage, both can now exceed
# that limit outright - observed live as psycopg.OperationalError ("number
# of parameters must be between 0 and 65535") on a real 4-event bbox (a
# single per-anchor OSM fetch alone was seen returning 29,000+ nodes, i.e.
# 87,000+ params for an unbatched 3-column VALUES insert). 10,000 is
# comfortably under the limit for both the 1-param-per-row IN(...) lookup
# and the 3-param-per-row node upsert (30,000 params/batch).
_BULK_OPERATION_BATCH_SIZE = 10_000


class RoadNetworkRepository:
    """Persists and retrieves the road network graph via SQLAlchemy."""

    def save_network(self, db: Session, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        """Upsert road-network nodes and bulk-insert edges into the database.

        Task A1.5: previously used `Session.merge()` in a per-object Python
        loop. `merge()` with a primary key already set (every node carries
        its real OSM node id) issues a synchronous SELECT-by-primary-key
        round trip *per object* to decide insert vs. update. For a real
        regional road network - tens of thousands of nodes/edges from one
        OSM fetch - against a remote database, that is many minutes of pure
        per-row network latency, not an OSM/network-fetch problem (this was
        the dominant contributor to the observed multi-minute simulation
        stall, on top of the separately-bounded OSM fetch itself - see
        RoadNetworkFetcher.OSM_FETCH_TIMEOUT_SECONDS).

        Nodes now use one batched `INSERT ... ON CONFLICT DO UPDATE`
        statement (dialect-aware: PostgreSQL in production, SQLite in
        tests) so a node already stored from a previous, overlapping bbox
        fetch gets its coordinates refreshed instead of raising a
        primary-key violation - preserving the original re-import
        semantics, just in one round trip instead of one per node.

        Edges never carry a pre-assigned id here - a fresh OSM fetch always
        builds `GraphEdge` without one (`RoadNetworkFetcher._build_edges`),
        since `GraphEdgeDB.id` is a surrogate autoincrement key, not an OSM
        id - so they only need a single batched INSERT, matching the
        previous per-edge insert-via-merge behavior exactly.

        Nodes are saved before edges so that, within this same unit of
        work, edge foreign keys always resolve to a node that is being (or
        already was) persisted. Commits once at the end, same as before.
        """
        if nodes:
            self._upsert_nodes(db, nodes)
        if edges:
            self._insert_edges(db, edges)
        db.commit()

    def _upsert_nodes(self, db: Session, nodes: list[GraphNode]) -> None:
        values = [{"id": node.id, "latitude": node.latitude, "longitude": node.longitude} for node in nodes]
        # Batched for the same reason as _edges_touching: a single combined
        # multi-row VALUES INSERT passes 3 bind parameters per node in ONE
        # statement, which can exceed PostgreSQL's 65535-parameter limit on
        # its own for a large OSM fetch (see _BULK_OPERATION_BATCH_SIZE).
        for start in range(0, len(values), _BULK_OPERATION_BATCH_SIZE):
            chunk = values[start : start + _BULK_OPERATION_BATCH_SIZE]
            insert_stmt = self._dialect_insert(db)(GraphNodeDB).values(chunk)
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=[GraphNodeDB.id],
                set_={
                    "latitude": insert_stmt.excluded.latitude,
                    "longitude": insert_stmt.excluded.longitude,
                },
            )
            db.execute(upsert_stmt)

    def _insert_edges(self, db: Session, edges: list[GraphEdge]) -> None:
        values = [
            {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "distance_meters": edge.distance_meters,
                "travel_time_seconds": edge.travel_time_seconds,
            }
            for edge in edges
        ]
        db.execute(insert(GraphEdgeDB), values)

    @staticmethod
    def _dialect_insert(db: Session):
        """Return the dialect-specific `insert()` supporting `on_conflict_do_update`."""
        dialect_name = db.get_bind().dialect.name
        if dialect_name == "sqlite":
            return sqlite.insert
        return postgresql.insert

    def get_network_in_bbox(
        self,
        db: Session,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return the road-network subgraph strictly within a bounding box.

        Nodes are matched where `min_lat < latitude < max_lat` and
        `min_lon < longitude < max_lon`. Edges are matched only when BOTH
        their source and target node fall within that same node set, so the
        returned edges never reference a node outside the returned node
        list (no edges "leading to nowhere").
        """
        self._validate_bbox(min_lat, max_lat, min_lon, max_lon)

        db_nodes = (
            db.execute(
                select(GraphNodeDB).where(
                    GraphNodeDB.latitude > min_lat,
                    GraphNodeDB.latitude < max_lat,
                    GraphNodeDB.longitude > min_lon,
                    GraphNodeDB.longitude < max_lon,
                )
            )
            .scalars()
            .all()
        )
        node_ids = {db_node.id for db_node in db_nodes}

        db_edges = self._edges_touching(db, node_ids)

        nodes = [GraphNode.model_validate(db_node) for db_node in db_nodes]
        edges = [GraphEdge.model_validate(db_edge) for db_edge in db_edges]
        return nodes, edges

    @staticmethod
    def _edges_touching(db: Session, node_ids: set[int]) -> list[GraphEdgeDB]:
        """Return every GraphEdgeDB whose source AND target are both in
        `node_ids`, batching the lookup to stay under PostgreSQL's 65535
        bind-parameter limit (see _BULK_OPERATION_BATCH_SIZE).

        Only `source_node_id` is filtered in SQL, in chunks; `target_node_id`
        membership is checked in Python against the already-materialized
        `node_ids` set instead of a second chunked IN(...) clause - same
        result as the original single two-clause query (an edge is only
        included when BOTH endpoints are in node_ids), one query dimension
        instead of a source x target cross-product of chunks.
        """
        if not node_ids:
            return []

        node_id_list = list(node_ids)
        edges: list[GraphEdgeDB] = []
        for start in range(0, len(node_id_list), _BULK_OPERATION_BATCH_SIZE):
            chunk = node_id_list[start : start + _BULK_OPERATION_BATCH_SIZE]
            chunk_edges = (
                db.execute(select(GraphEdgeDB).where(GraphEdgeDB.source_node_id.in_(chunk)))
                .scalars()
                .all()
            )
            edges.extend(edge for edge in chunk_edges if edge.target_node_id in node_ids)
        return edges

    @staticmethod
    def _validate_bbox(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> None:
        for field_name, value in (
            ("min_lat", min_lat),
            ("max_lat", max_lat),
            ("min_lon", min_lon),
            ("max_lon", max_lon),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number, got {value!r}.")
        if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90:
            raise ValueError(f"latitude bounds must be within [-90, 90], got min_lat={min_lat!r}, max_lat={max_lat!r}.")
        if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180:
            raise ValueError(
                f"longitude bounds must be within [-180, 180], got min_lon={min_lon!r}, max_lon={max_lon!r}."
            )
        if min_lat >= max_lat:
            raise ValueError(f"min_lat must be less than max_lat, got min_lat={min_lat!r}, max_lat={max_lat!r}.")
        if min_lon >= max_lon:
            raise ValueError(f"min_lon must be less than max_lon, got min_lon={min_lon!r}, max_lon={max_lon!r}.")
