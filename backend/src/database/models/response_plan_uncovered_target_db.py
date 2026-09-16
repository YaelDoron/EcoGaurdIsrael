"""SQLAlchemy ORM model for uncovered response-plan target ids."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_plan_db import ResponsePlanDB


class ResponsePlanUncoveredTargetDB(Base):
    """One ordered uncovered target id in a response plan."""

    __tablename__ = "response_plan_uncovered_targets"
    __table_args__ = (
        UniqueConstraint("response_plan_id", "target_order", name="uq_response_plan_uncovered_order"),
        UniqueConstraint("response_plan_id", "response_target_id", name="uq_response_plan_uncovered_target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False,
        index=True,
    )
    target_order: Mapped[int] = mapped_column(Integer, nullable=False)
    response_target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    response_plan: Mapped["ResponsePlanDB"] = relationship(back_populates="uncovered_targets")
