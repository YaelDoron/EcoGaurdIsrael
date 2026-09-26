"""SQLAlchemy ORM model for one FireEvent's latest runtime ML/decision trace (Task 5).

Entirely additive: a new table, one row per FireEvent (upserted, not a
history log), with a unique FK to fire_events. Historical FireEvents simply
have no matching row here - never backfilled, never a fabricated "ML not
evaluated" row inserted retroactively. FireEventDB itself is unchanged.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, deferred, mapped_column

from src.database.base import Base

# Deferred-load group of the Task 9B ai_hybrid_v5 audit columns (see FireEventMLAssessmentDB).
AI_HYBRID_V5_COLUMN_GROUP = "ai_hybrid_v5_audit"


class FireEventMLAssessmentDB(Base):
    """The single latest ML/decision trace row for one FireEvent."""

    __tablename__ = "fire_event_ml_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    decision_mode: Mapped[str] = mapped_column(String, nullable=False)
    rule_status: Mapped[str] = mapped_column(String, nullable=False)
    rule_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    ml_available: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ml_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ml_model_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ml_model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ml_feature_schema_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ml_failure_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    agreement: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Task 9B (AI_HYBRID_V5 audit). Additive, nullable and DEFERRED: a SELECT of this row does not reference them unless
    # they are read, and the repository writes them only for ai_hybrid_v5 rows - so RULE_ONLY / SHADOW / HYBRID never touch
    # these columns and keep working on a database that has not been migrated yet
    # (scripts/migrate_add_fire_event_ml_assessment_ai_columns.py adds them; only ai_hybrid_v5 needs it).
    # One deferred GROUP, so an ai_hybrid_v5 read loads all five with the row (undefer_group) instead of one
    # lazy round trip per column per row.
    policy_version: Mapped[Optional[str]] = deferred(
        mapped_column(String, nullable=True), group=AI_HYBRID_V5_COLUMN_GROUP
    )
    policy_status: Mapped[Optional[str]] = deferred(
        mapped_column(String, nullable=True), group=AI_HYBRID_V5_COLUMN_GROUP
    )
    history_available: Mapped[Optional[bool]] = deferred(
        mapped_column(Boolean, nullable=True), group=AI_HYBRID_V5_COLUMN_GROUP
    )
    satellite_pass_count: Mapped[Optional[int]] = deferred(
        mapped_column(Integer, nullable=True), group=AI_HYBRID_V5_COLUMN_GROUP
    )
    current_satellite_pixel_count: Mapped[Optional[int]] = deferred(
        mapped_column(Integer, nullable=True), group=AI_HYBRID_V5_COLUMN_GROUP
    )
