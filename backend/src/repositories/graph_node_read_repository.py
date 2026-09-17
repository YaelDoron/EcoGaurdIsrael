"""Read-only bulk lookup of persisted GraphNode coordinates (Epic 6, US 6.3, Task 4).

`RoadNetworkRepository` (Epic 5) is built around OSM import and bounding-box
subgraph queries for pathfinding - it has no "give me exactly these node ids"
lookup. `GraphNodeReadRepository` is a small, US 6.3-owned, read-only
sibling for exactly that: turning a `RouteResult.node_path` (a tuple of
already-persisted `GraphNodeDB` ids) into coordinates for display, in one
bulk query rather than one query per node. It performs no OSM fetching, no
bounding-box lookup, no graph construction, no routing, and no writes.
"""
from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.graph_node_db import GraphNodeDB


class GraphNodeReadRepository:
    """Bulk read-only access to persisted `GraphNodeDB` rows by id."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    def get_by_ids(self, node_ids: Iterable[int]) -> dict[int, GraphNodeDB]:
        """Return persisted GraphNodeDB rows for the given ids, keyed by id.

        Issues a single `WHERE id IN (...)` query regardless of how many ids
        are requested. An id with no matching row is simply absent from the
        returned dict - callers must not invent a coordinate for it.
        """
        node_ids = list(dict.fromkeys(node_ids))
        if not node_ids:
            return {}

        session = self._session_factory()
        try:
            rows = session.execute(select(GraphNodeDB).where(GraphNodeDB.id.in_(node_ids))).scalars().all()
            return {row.id: row for row in rows}
        finally:
            session.close()
