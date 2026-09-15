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

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode


class RoadNetworkRepository:
    """Persists and retrieves the road network graph via SQLAlchemy."""

    def save_network(self, db: Session, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        """Upsert road-network nodes and edges into the database.

        Uses `Session.merge()` rather than `Session.add()` so re-importing
        OSM data - where a node or edge id may already exist from a
        previous import - updates the existing row in place instead of
        raising a primary-key/unique-constraint violation.

        Nodes are merged before edges so that, within this same unit of
        work, edge foreign keys always resolve to a node that is being
        (or already was) persisted. Commits once at the end.
        """
        for node in nodes:
            db.merge(GraphNodeDB(id=node.id, latitude=node.latitude, longitude=node.longitude))

        for edge in edges:
            db.merge(
                GraphEdgeDB(
                    id=edge.id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    distance_meters=edge.distance_meters,
                    travel_time_seconds=edge.travel_time_seconds,
                )
            )

        db.commit()

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

        db_edges: list[GraphEdgeDB] = []
        if node_ids:
            db_edges = (
                db.execute(
                    select(GraphEdgeDB).where(
                        GraphEdgeDB.source_node_id.in_(node_ids),
                        GraphEdgeDB.target_node_id.in_(node_ids),
                    )
                )
                .scalars()
                .all()
            )

        nodes = [GraphNode.model_validate(db_node) for db_node in db_nodes]
        edges = [GraphEdge.model_validate(db_edge) for db_edge in db_edges]
        return nodes, edges

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
