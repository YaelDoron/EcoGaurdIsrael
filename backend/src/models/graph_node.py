"""Pydantic logic model for a road-network node (intersection or waypoint).

Mirrors the persistence-layer `src.database.models.graph_node_db.GraphNodeDB`.
`from_attributes=True` lets this model be built directly from a GraphNodeDB
ORM instance, e.g. `GraphNode.model_validate(db_node)`.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GraphNode(BaseModel):
    """A road-network node, identified by its OpenStreetMap (OSM) node id."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    latitude: float
    longitude: float
