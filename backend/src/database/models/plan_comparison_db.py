"""SQLAlchemy ORM model for one persisted optimized-vs-baseline PlanComparison.

Implements User Story 5.3 Task 4: persistence for Task 3's `PlanComparison`
domain result. Append-only history -- `PlanComparisonRepository.save(...)`
always inserts a new row and never updates an existing one, mirroring
`FireSpreadPredictionDB`/`ResponseTargetSetDB`.

`fire_event_id`, `optimized_plan_id`, `route_planning_run_id`, and
`response_target_set_id` are the exact planning-snapshot identities the
`PlanComparison` was computed from (see Task 3's snapshot-consistency
check). They are intentionally plain, non-FK integer columns for now:
`FireEventDB` and `ResponseTargetSetDB` already exist in this codebase, but
`optimized_plan_id` / `route_planning_run_id` reference Company 1/2's
`ResponsePlan` / `RoutePlanningRun` tables, which do not exist yet. Rather
than add FK constraints to only two of the four snapshot identities, all
four stay plain scalar traceability columns for consistency -- Task 4 does
not perform cross-table validation (see Task 5, which owns loading/
validating the real planning-chain records). FK constraints for all four
can be reconsidered together once Company 1/2's persistence is merged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class PlanComparisonDB(Base):
    """One persisted, immutable optimized-vs-baseline comparison row."""

    __tablename__ = "plan_comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    fire_event_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    optimized_plan_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    route_planning_run_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    response_target_set_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

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
