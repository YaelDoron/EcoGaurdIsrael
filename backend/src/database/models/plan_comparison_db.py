"""SQLAlchemy ORM model for one persisted optimized-vs-baseline PlanComparison.

Implements User Story 5.3 Task 4: persistence for Task 3's `PlanComparison`
domain result. Append-only history -- `PlanComparisonRepository.save(...)`
always inserts a new row and never updates an existing one, mirroring
`FireSpreadPredictionDB`/`ResponseTargetSetDB`.

`fire_event_id`, `optimized_plan_id`, `route_planning_run_id`, and
`response_target_set_id` are the exact planning-snapshot identities the
`PlanComparison` was computed from (see Task 3's snapshot-consistency
check). They were originally left as plain, non-FK integer columns because
`optimized_plan_id` / `route_planning_run_id` referenced Company 1/2's
`ResponsePlan` / `RoutePlanningRun` tables, which did not exist yet at the
time (all four were kept FK-free together for consistency rather than only
two of them). That persistence has since been merged (`ResponsePlanDB`,
`RoutePlanningRunDB`), so all four now carry explicit FK constraints
(FND-05). This does not add cross-table validation logic here -- Task 5
still owns loading/validating the real planning-chain records; the FKs only
guarantee referential integrity at the database level.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class PlanComparisonDB(Base):
    """One persisted, immutable optimized-vs-baseline comparison row."""

    __tablename__ = "plan_comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    optimized_plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False,
        index=True,
    )
    route_planning_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("route_planning_runs.id"),
        nullable=False,
        index=True,
    )
    response_target_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_target_sets.id"),
        nullable=False,
        index=True,
    )

    optimized_score: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_score: Mapped[float] = mapped_column(Float, nullable=False)
    optimized_coverage_score: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_coverage_score: Mapped[float] = mapped_column(Float, nullable=False)
    optimized_average_eta_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_average_eta_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_difference: Mapped[float] = mapped_column(Float, nullable=False)
    improvement_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
