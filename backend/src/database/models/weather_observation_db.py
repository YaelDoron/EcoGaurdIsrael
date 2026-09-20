"""SQLAlchemy ORM model for the weather_observations table.

This is the persistence-layer counterpart of the Task 2 domain dataclass
`src.models.weather_observation.WeatherObservation`. The two are kept
intentionally separate - `WeatherRepository` is the only place that converts
between them.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.weather_station_db import WeatherStationDB


class WeatherObservationDB(Base):
    """A stored weather observation, linked to its station via the internal DB id.

    A unique constraint on (station_id, timestamp) prevents duplicate
    observations at the database level - this is not only a Python-side
    check, PostgreSQL itself enforces it.
    """

    __tablename__ = "weather_observations"
    __table_args__ = (
        UniqueConstraint("station_id", "timestamp", name="uq_weather_observations_station_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weather_stations.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relative_humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_gust: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rainfall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    station: Mapped["WeatherStationDB"] = relationship(back_populates="observations")
