"""SQLAlchemy ORM model for the weather_stations table.

This is the persistence-layer counterpart of the Task 2 domain dataclass
`src.models.weather_station.WeatherStation`. The two are kept intentionally
separate - `WeatherRepository` is the only place that converts between them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.weather_observation_db import WeatherObservationDB


class WeatherStationDB(Base):
    """A stored weather station.

    `id` is EcoGuard's own internal database id. `external_station_id` is
    the IMS station id - the two must never be confused.
    """

    __tablename__ = "weather_stations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_station_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    region_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # No delete-orphan / delete cascade: weather history must survive station
    # metadata updates and must not be silently deleted.
    observations: Mapped[list["WeatherObservationDB"]] = relationship(
        back_populates="station",
        cascade="save-update, merge",
    )
