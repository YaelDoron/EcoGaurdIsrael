"""SQLAlchemy ORM model for one persisted routing run (Epic 5 / User Story 5.1)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.route_result_db import RouteResultDB


class RoutePlanningRunDB(Base):
    """Append-only record of one routing pass over a response-target set's resources/targets."""

    __tablename__ = "route_planning_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    response_target_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_target_sets.id"),
        nullable=False,
        index=True,
    )
    planned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    # A flat list of the resource ids considered by this run - bookkeeping
    # about the planning context, not a set of independently queryable
    # entities, so a JSON column is used rather than a junction table (the
    # actual per-pairing query surface is RouteResultDB.resource_id).
    resource_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    routes: Mapped[list["RouteResultDB"]] = relationship(
        back_populates="planning_run",
        cascade="all, delete-orphan",
    )
