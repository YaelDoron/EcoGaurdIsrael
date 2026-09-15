"""SQLAlchemy ORM model for the graph_nodes table (road network intersections).

This is the persistence-layer counterpart of the domain model
`src.models.graph_node.GraphNode`. `id` is the OpenStreetMap (OSM) node id,
used directly as the primary key rather than a surrogate database id, so
stored nodes stay directly cross-referenceable with OSM source data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.graph_edge_db import GraphEdgeDB


class GraphNodeDB(Base):
    """A stored road-network node (an intersection or waypoint), keyed by OSM node id."""

    __tablename__ = "graph_nodes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    outgoing_edges: Mapped[list["GraphEdgeDB"]] = relationship(
        back_populates="source_node",
        foreign_keys="GraphEdgeDB.source_node_id",
        cascade="save-update, merge",
    )
    incoming_edges: Mapped[list["GraphEdgeDB"]] = relationship(
        back_populates="target_node",
        foreign_keys="GraphEdgeDB.target_node_id",
        cascade="save-update, merge",
    )
