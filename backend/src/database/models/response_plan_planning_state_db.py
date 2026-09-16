"""SQLAlchemy ORM model linking a ResponsePlan to the PlanningEffectiveState
fingerprint that produced it (Epic 5, User Story 5.4, Task 4).

This is a US 5.4-owned sidecar, not a change to Company 2's ResponsePlan
persistence. It exists only to record, for one already-persisted
ResponsePlan, which semantic PlanningEffectiveState fingerprint (Task 3)
produced it - append-only, at most one row per ResponsePlan.

FireEvent, ResponseTargetSet, RoutePlanningRun, and resource identities are
deliberately NOT duplicated here: they are already reachable via
`response_plan_id -> ResponsePlan -> {response_target_set_id,
route_planning_run_id -> resource_ids}`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class ResponsePlanPlanningStateDB(Base):
    """Append-only 1:1 link from one ResponsePlan to the fingerprint that produced it."""

    __tablename__ = "response_plan_planning_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    planning_effective_state_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
