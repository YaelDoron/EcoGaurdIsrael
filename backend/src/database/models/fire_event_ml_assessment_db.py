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
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


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
