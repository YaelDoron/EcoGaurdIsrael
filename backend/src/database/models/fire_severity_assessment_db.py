"""SQLAlchemy ORM model for persisted fire-severity assessments."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_severity_assessment_satellite_input_db import (
        FireSeverityAssessmentSatelliteInputDB,
    )
    from src.database.models.fire_severity_assessment_weather_input_db import (
        FireSeverityAssessmentWeatherInputDB,
    )


class FireSeverityAssessmentDB(Base):
    """A persisted active-fire severity assessment for one FireEvent."""

    __tablename__ = "fire_severity_assessments"
    __table_args__ = (
        CheckConstraint(
            "("
            "status = 'valid' AND score IS NOT NULL AND severity_level IS NOT NULL"
            ") OR ("
            "status IN ('insufficient_data', 'inactive_event') "
            "AND score IS NULL AND severity_level IS NULL"
            ")",
            name="ck_fire_severity_assessments_status_result_nullability",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity_level: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)

    vegetation_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    vegetation_dataset_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vegetation_radius_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vegetation_dominant_land_cover: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    vegetation_fuel_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    weather_inputs: Mapped[list["FireSeverityAssessmentWeatherInputDB"]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
    )
    satellite_inputs: Mapped[list["FireSeverityAssessmentSatelliteInputDB"]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
    )
