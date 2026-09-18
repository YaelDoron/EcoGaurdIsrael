"""SQLAlchemy ORM model for one FireEvent's membership row within a
GlobalPlanningRun (Stage 2 of the Global Multi-Incident Optimizer
refactor). See src/models/global_planning_run_event.py for the domain
model this persists.

`response_plan_id` intentionally has NO FK-level requirement to be the
newly created plan for this run - see ResponsePlanDB.global_planning_run_id
and GlobalPlanningOrchestrator's own docstring for why "which run observed
this event" (this table) and "which run created this ResponsePlan"
(response_plans.global_planning_run_id) are different questions that can
disagree (e.g. a NO_OP member observed a plan created by an EARLIER run).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class GlobalPlanningRunEventDB(Base):
    """One FireEvent's membership + recorded outcome within one GlobalPlanningRun.

    The severity/demand snapshot columns below (Stage 6, demand/shortage
    historical persistence) are this FireEvent's exact severity and
    demand-vs-supply accounting AS OF this run - populated only for a
    member whose cycle actually reached the GA (see
    GlobalPlanningRunRepository.record_member_result's `incident_demand`/
    `event_optimization_result` parameters). A NO_OP member's semantic
    state is unchanged from the prior generation, so these are left NULL
    rather than re-persisting a duplicate snapshot.
    """

    __tablename__ = "global_planning_run_events"
    __table_args__ = (
        UniqueConstraint("global_planning_run_id", "fire_event_id", name="uq_global_planning_run_events_run_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_planning_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("global_planning_runs.id"),
        nullable=False,
        index=True,
    )
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    event_order: Mapped[int] = mapped_column(Integer, nullable=False)
    result_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    response_plan_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=True,
        index=True,
    )
    local_state_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    severity_assessment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    severity_level: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    severity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    demand_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    demand_policy_methodology: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    demand_policy_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    minimum_resources: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    desired_resources: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    assigned_resources: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    required_slots_covered: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    required_slots_uncovered: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    desired_slots_covered: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    desired_slots_uncovered: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_risk_slots_covered: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    locked_resources_preserved: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    coverage_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_eta_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
