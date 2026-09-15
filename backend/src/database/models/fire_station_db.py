"""SQLAlchemy ORM model for the fire_stations table.

This is the persistence-layer counterpart of the domain dataclass
`src.models.fire_station.FireStation`. `id` is the station identifier as
reported by the governmental source data (stored as text since that source
may use non-numeric ids), not a surrogate database id.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.firefighting_resource_db import FirefightingResourceDB


class FireStationDB(Base):
    """A stored fire station."""

    __tablename__ = "fire_stations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    station_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    address: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    resources: Mapped[list["FirefightingResourceDB"]] = relationship(
        back_populates="station",
        cascade="save-update, merge",
    )
