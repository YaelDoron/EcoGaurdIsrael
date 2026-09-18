"""SQLAlchemy ORM model for one global planning cycle (Stage 2 of the Global
Multi-Incident Optimizer refactor). See src/models/global_planning_run.py
for the domain model this persists.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class GlobalPlanningRunDB(Base):
    """One global planning cycle: RUNNING -> COMPLETED/PARTIAL/FAILED/NO_ACTIVE_EVENTS.

    The optimization-metadata and shortage columns below (Stage 6, demand/
    shortage historical persistence) are the exact snapshot of one
    successfully-activated (or partially-activated) cycle's global
    optimization outcome - nullable and populated exactly once, by
    GlobalPlanningRunRepository.record_global_optimization_metadata, right
    before the run is completed. They are never recomputed from current
    state later: a NO_OP/NO_ACTIVE_EVENTS/pre-GA-FAILED run leaves them
    NULL rather than duplicating the previous generation's values.
    """

    __tablename__ = "global_planning_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    trigger: Mapped[str] = mapped_column(String, nullable=False)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    input_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Stage 6: raw GlobalPlanningInput fingerprint and the optimization
    # policy fingerprint, kept distinct from `input_fingerprint` above
    # (which Stage 6 repurposed to hold their COMBINED value - see
    # GlobalPlanningRefreshCoordinator._combined_fingerprint).
    raw_input_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    optimization_policy_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    random_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ga_population_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ga_generation_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ga_mutation_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ga_crossover_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    demand_scoring_policy_methodology: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    demand_scoring_policy_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    severity_demand_policy_methodology: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    severity_demand_policy_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stability_policy_methodology: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stability_policy_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    fitness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coverage_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_eta_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # GlobalResourceShortage snapshot (Stage 5) - the aggregate demand-vs-supply
    # picture across every active FireEvent in this cycle.
    shortage_total_required: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_total_desired: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_total_assigned: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_unmet_required: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_unmet_desired: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_candidate_assignable_resource_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_committed_resource_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_unavailable_resource_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shortage_locked_resources_preserved: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
