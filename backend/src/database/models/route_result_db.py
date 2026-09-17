"""SQLAlchemy ORM model for one persisted RouteResult inside a RoutePlanningRun."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, BigInteger, CheckConstraint, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.route_planning_run_db import RoutePlanningRunDB


class RouteResultDB(Base):
    """One resource-to-target routing outcome inside a persisted RoutePlanningRun."""

    __tablename__ = "route_results"
    __table_args__ = (
        UniqueConstraint(
            "route_planning_run_id",
            "resource_id",
            "response_target_id",
            name="uq_route_results_run_resource_target",
        ),
        CheckConstraint(
            "("
            "status = 'reachable' "
            "AND source_node_id IS NOT NULL AND target_node_id IS NOT NULL "
            "AND distance_meters IS NOT NULL AND distance_meters >= 0 "
            "AND travel_time_seconds IS NOT NULL AND travel_time_seconds >= 0"
            ") OR ("
            "status = 'unreachable' "
            "AND source_node_id IS NOT NULL AND target_node_id IS NOT NULL "
            "AND distance_meters IS NULL AND travel_time_seconds IS NULL"
            ") OR ("
            "status = 'unmappable' "
            "AND NOT (source_node_id IS NOT NULL AND target_node_id IS NOT NULL) "
            "AND distance_meters IS NULL AND travel_time_seconds IS NULL"
            ")",
            name="ck_route_results_status_field_consistency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_planning_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("route_planning_runs.id"),
        nullable=False,
        index=True,
    )
    resource_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("firefighting_resources.id"),
        nullable=False,
        index=True,
    )
    response_target_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_targets.id"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String, nullable=False)

    source_node_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("graph_nodes.id"), nullable=True
    )
    target_node_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("graph_nodes.id"), nullable=True
    )
    node_path: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    distance_meters: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    travel_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    planning_run: Mapped["RoutePlanningRunDB"] = relationship(back_populates="routes")
