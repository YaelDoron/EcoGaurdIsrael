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

# Regression this guards against: a bbox spanning multiple active incidents
# over the (intentionally never-purged, ever-growing) road-network cache can
# select tens of thousands of nodes. Filtering edges with
# `source_node_id.in_(node_ids)` AND `target_node_id.in_(node_ids)` in one
# query binds ~2x that many parameters, which exceeds PostgreSQL's ~65,535
# per-statement parameter limit long before the node count itself becomes
# unreasonable. Only `source_node_id` is chunked into an `.in_()` clause
# here; the target-node check is done in Python against a `set`, so no
# query ever binds more than EDGE_QUERY_CHUNK_SIZE parameters, regardless of
# how large the cached graph or the requested bbox grows.
EDGE_QUERY_CHUNK_SIZE = 2000


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

        Edges are looked up in `EDGE_QUERY_CHUNK_SIZE`-sized chunks of
        `source_node_id` only (never a joint source+target `.in_()` query -
        see EDGE_QUERY_CHUNK_SIZE's docstring), with the target-node check
        done in Python against a `set`. This keeps every query's parameter
        count bounded regardless of how large the requested bbox or the
        cached graph is, and never risks missing a "cross-chunk" edge: an
        edge is found by its source's chunk alone, and its target is then
        checked against the *complete* node-id set, not the same chunk.
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

        db_edges = self._get_edges_within_node_set(db, node_ids)

        nodes = [GraphNode.model_validate(db_node) for db_node in db_nodes]
        edges = [GraphEdge.model_validate(db_edge) for db_edge in db_edges]
        return nodes, edges

    @staticmethod
    def _get_edges_within_node_set(db: Session, node_ids: set[int]) -> list[GraphEdgeDB]:
        """Return every edge whose source AND target are both in `node_ids`.

        Queries `source_node_id` in bounded chunks (never both endpoints in
        one `.in_()` query - see EDGE_QUERY_CHUNK_SIZE) and filters the
        target endpoint in Python against `node_ids` (a `set`, so each
        membership check is O(1); never re-queries the database per edge).
        Deduplicates by primary key, since a node can only appear in one
        chunk, so this is defensive rather than load-bearing today.
        """
        if not node_ids:
            return []

        node_id_list = list(node_ids)
        edges_by_id: dict[int, GraphEdgeDB] = {}
        for start in range(0, len(node_id_list), EDGE_QUERY_CHUNK_SIZE):
            chunk = node_id_list[start : start + EDGE_QUERY_CHUNK_SIZE]
            chunk_edges = (
                db.execute(select(GraphEdgeDB).where(GraphEdgeDB.source_node_id.in_(chunk))).scalars().all()
            )
            for edge in chunk_edges:
                if edge.target_node_id in node_ids:
                    edges_by_id[edge.id] = edge

        return list(edges_by_id.values())

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
