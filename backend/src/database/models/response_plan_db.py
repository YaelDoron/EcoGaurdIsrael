"""SQLAlchemy ORM model for persisted response optimization plans."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_action_db import ResponseActionDB
    from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB

_OPTIMIZATION_CONFIG_COLUMNS = (
    "population_size",
    "generation_count",
    "mutation_rate",
    "crossover_rate",
    "eta_reference_seconds",
    "initial_assignment_probability",
    "tournament_size",
    "elitism_count",
)


class ResponsePlanDB(Base):
    """Append-only generated response-plan recommendation.

    `random_seed` (always persisted) plus the 8 `_OPTIMIZATION_CONFIG_COLUMNS`
    (nullable, all-or-nothing - see the CheckConstraint below) together
    capture the exact resolved ResponseOptimizationConfig that produced the
    plan (FND-04). Legacy rows predating this capture have all 8 columns
    NULL: "configuration unavailable", never fabricated from current
    source-code defaults.
    """

    __tablename__ = "response_plans"
    __table_args__ = (
        CheckConstraint(
            "(" + " AND ".join(f"{column} IS NULL" for column in _OPTIMIZATION_CONFIG_COLUMNS) + ")"
            " OR ("
            + " AND ".join(f"{column} IS NOT NULL" for column in _OPTIMIZATION_CONFIG_COLUMNS)
            + ")",
            name="ck_response_plans_optimization_config_all_or_none",
        ),
    )

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
    route_planning_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("route_planning_runs.id"),
        nullable=False,
        index=True,
    )
    # Stage 2 (Global Multi-Incident Optimizer refactor): which GlobalPlanningRun
    # CREATED this plan, if any. Nullable for every historical plan (no backfill -
    # see scripts/migrate_global_planning_run.py) and for any plan created outside
    # a global cycle. Never set for a NO_OP/baseline-only-recovery result that
    # reuses an already-existing plan - see GlobalPlanningOrchestrator.
    global_planning_run_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("global_planning_runs.id"),
        nullable=True,
        index=True,
    )
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    plan_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coverage_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_eta_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Exact GA configuration snapshot (FND-04). Nullable/all-or-nothing: see
    # ck_response_plans_optimization_config_all_or_none above.
    population_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generation_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mutation_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    crossover_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eta_reference_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    initial_assignment_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tournament_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    elitism_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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
