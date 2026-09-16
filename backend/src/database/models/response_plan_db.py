"""SQLAlchemy ORM model for persisted response optimization plans."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_action_db import ResponseActionDB
    from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB


class ResponsePlanDB(Base):
    """Append-only generated response-plan recommendation."""

    __tablename__ = "response_plans"

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
    route_planning_run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coverage_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_eta_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    actions: Mapped[list["ResponseActionDB"]] = relationship(
        back_populates="response_plan",
        cascade="all, delete-orphan",
    )
    uncovered_targets: Mapped[list["ResponsePlanUncoveredTargetDB"]] = relationship(
        back_populates="response_plan",
        cascade="all, delete-orphan",
    )
