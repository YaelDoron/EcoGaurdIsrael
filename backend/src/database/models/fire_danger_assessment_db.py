"""SQLAlchemy ORM model for persisted fire-danger assessments."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_danger_assessment_weather_input_db import (
        FireDangerAssessmentWeatherInputDB,
    )


class FireDangerAssessmentDB(Base):
    """A persisted fire-danger assessment for one area and assessment time."""

    __tablename__ = "fire_danger_assessments"
    __table_args__ = (
        CheckConstraint(
            "("
            "status = 'valid' AND score IS NOT NULL AND danger_level IS NOT NULL"
            ") OR ("
            "status = 'insufficient_data' AND score IS NULL AND danger_level IS NULL"
            ")",
            name="ck_fire_danger_assessments_status_result_nullability",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    area_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    area_name: Mapped[str] = mapped_column(String, nullable=False)
    area_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    area_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    area_radius_km: Mapped[float] = mapped_column(Float, nullable=False)

    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    danger_level: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    weather_inputs: Mapped[list["FireDangerAssessmentWeatherInputDB"]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
    )
