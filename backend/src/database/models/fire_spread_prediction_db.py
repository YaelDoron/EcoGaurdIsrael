"""SQLAlchemy ORM model for persisted wildfire-spread prediction runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
    from src.database.models.fire_spread_prediction_weather_input_db import (
        FireSpreadPredictionWeatherInputDB,
    )


class FireSpreadPredictionDB(Base):
    """A persisted deterministic wildfire-spread prediction run for one FireEvent.

    Append-only history: a new row is inserted per prediction run, never
    updated in place -- mirrors FireDangerAssessmentDB/FireSeverityAssessmentDB.
    """

    __tablename__ = "fire_spread_predictions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('valid', 'insufficient_data', 'inactive_event')",
            name="ck_fire_spread_predictions_status_values",
        ),
        CheckConstraint(
            "(status = 'valid' AND severity_assessment_id IS NOT NULL) OR (status != 'valid')",
            name="ck_fire_spread_predictions_valid_requires_severity_assessment",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    severity_assessment_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("fire_severity_assessments.id"),
        nullable=True,
        index=True,
    )
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    effective_state_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    cells: Mapped[list["FireSpreadPredictionCellDB"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
    )
    weather_inputs: Mapped[list["FireSpreadPredictionWeatherInputDB"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
    )
