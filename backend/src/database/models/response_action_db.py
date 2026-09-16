"""SQLAlchemy ORM model for persisted response-plan actions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_plan_db import ResponsePlanDB


class ResponseActionDB(Base):
    """One ordered resource-to-target recommendation in a response plan."""

    __tablename__ = "response_actions"
    __table_args__ = (
        UniqueConstraint("response_plan_id", "action_order", name="uq_response_actions_plan_order"),
        UniqueConstraint("response_plan_id", "resource_id", name="uq_response_actions_plan_resource"),
        UniqueConstraint("response_plan_id", "response_target_id", name="uq_response_actions_plan_target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False,
        index=True,
    )
    action_order: Mapped[int] = mapped_column(Integer, nullable=False)
    resource_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    response_target_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    route_result_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    response_plan: Mapped["ResponsePlanDB"] = relationship(back_populates="actions")
