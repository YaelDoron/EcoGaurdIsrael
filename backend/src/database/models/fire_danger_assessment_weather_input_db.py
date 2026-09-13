"""SQLAlchemy ORM model linking fire-danger assessments to weather inputs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
    from src.database.models.weather_observation_db import WeatherObservationDB


class FireDangerAssessmentWeatherInputDB(Base):
    """Trace row for a weather observation used by a fire-danger assessment."""

    __tablename__ = "fire_danger_assessment_weather_inputs"
    __table_args__ = (
        UniqueConstraint(
            "assessment_id",
            "weather_observation_id",
            name="uq_fire_danger_assessment_weather_inputs_assessment_observation",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assessment_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_danger_assessments.id"),
        nullable=False,
        index=True,
    )
    weather_observation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("weather_observations.id"),
        nullable=False,
        index=True,
    )
    station_id: Mapped[int] = mapped_column(Integer, nullable=False)

    assessment: Mapped["FireDangerAssessmentDB"] = relationship(back_populates="weather_inputs")
    weather_observation: Mapped["WeatherObservationDB"] = relationship()
