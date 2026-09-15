"""Pydantic logic model for a directed road-network edge.

Mirrors the persistence-layer `src.database.models.graph_edge_db.GraphEdgeDB`.
`from_attributes=True` lets this model be built directly from a GraphEdgeDB
ORM instance, e.g. `GraphEdge.model_validate(db_edge)`. `id` is optional
since a not-yet-persisted edge has no surrogate database id assigned yet.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class GraphEdge(BaseModel):
    """A directed edge connecting two graph nodes in the road network."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    source_node_id: int
    target_node_id: int
    distance_meters: float
    travel_time_seconds: float
