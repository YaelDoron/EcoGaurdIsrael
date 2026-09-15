"""SQLAlchemy ORM model for the graph_edges table (road network segments).

This is the persistence-layer counterpart of the domain model
`src.models.graph_edge.GraphEdge`. Unlike GraphNodeDB, `id` here is a plain
surrogate database id: a single OSM way is typically split into several
directed routing edges between intersection nodes, so OSM way ids are not
1:1 with rows in this table.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.graph_node_db import GraphNodeDB


class GraphEdgeDB(Base):
    """A stored directed edge connecting two road-network nodes."""

    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_node_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("graph_nodes.id"), nullable=False, index=True
    )
    target_node_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("graph_nodes.id"), nullable=False, index=True
    )
    distance_meters: Mapped[float] = mapped_column(Float, nullable=False)
    travel_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    source_node: Mapped["GraphNodeDB"] = relationship(
        back_populates="outgoing_edges",
        foreign_keys=[source_node_id],
    )
    target_node: Mapped["GraphNodeDB"] = relationship(
        back_populates="incoming_edges",
        foreign_keys=[target_node_id],
    )
