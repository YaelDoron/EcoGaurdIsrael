"""SQLAlchemy ORM model linking a wildfire-spread prediction to its selected weather observation."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
    from src.database.models.weather_observation_db import WeatherObservationDB


class FireSpreadPredictionWeatherInputDB(Base):
    """Trace row for the weather observation selected by a spread prediction run.

    V1 always selects exactly one WeatherObservation per prediction (see
    FireSpreadInputService), so `prediction_id` is unique -- this prevents a
    prediction from accidentally acquiring more than one weather trace row.
    """

    __tablename__ = "fire_spread_prediction_weather_inputs"
    __table_args__ = (
        UniqueConstraint(
            "prediction_id",
            name="uq_fire_spread_weather_input_prediction",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_spread_predictions.id"),
        nullable=False,
        index=True,
    )
    weather_observation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("weather_observations.id"),
        nullable=False,
        index=True,
    )

    prediction: Mapped["FireSpreadPredictionDB"] = relationship(back_populates="weather_inputs")
    weather_observation: Mapped["WeatherObservationDB"] = relationship()
