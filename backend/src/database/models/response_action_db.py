"""SQLAlchemy ORM model for persisted response-plan actions."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_plan_db import ResponsePlanDB


class ResponseActionDB(Base):
    """One ordered resource-to-target recommendation in a response plan.

    No uniqueness constraint on (response_plan_id, response_target_id):
    Stage 5's demand-aware GlobalAllocationSlotFactory legitimately
    generates several allocation slots (slot_index 0, 1, 2, ...) against the
    SAME canonical ACTIVE_FIRE response_target_id whenever desired_resources
    > 1 (e.g. a HIGH/CRITICAL severity FireEvent with more than one
    suppression resource assigned) - a plan assigning two different
    resources to that one target is the correct, intended outcome, not a
    corruption. (response_plan_id, resource_id) below is still enough to
    prevent the real corruption this table must forbid: the same resource
    appearing twice in one plan.
    """

    __tablename__ = "response_actions"
    __table_args__ = (
        UniqueConstraint("response_plan_id", "action_order", name="uq_response_actions_plan_order"),
        UniqueConstraint("response_plan_id", "resource_id", name="uq_response_actions_plan_resource"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False,
        index=True,
    )
    action_order: Mapped[int] = mapped_column(Integer, nullable=False)
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
    route_result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("route_results.id"),
        nullable=False,
        index=True,
    )

    response_plan: Mapped["ResponsePlanDB"] = relationship(back_populates="actions")
