"""SQLAlchemy ORM model for the firefighting_resources table.

This is the persistence-layer counterpart of the domain dataclass
`src.models.firefighting_resource.FirefightingResource`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.models.resource_status import ResourceStatus

if TYPE_CHECKING:
    from src.database.models.fire_station_db import FireStationDB


class FirefightingResourceDB(Base):
    """A stored firefighting resource attached to a fire station."""

    __tablename__ = "firefighting_resources"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    station_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("fire_stations.id"),
        nullable=False,
        index=True,
    )
    status: Mapped[ResourceStatus] = mapped_column(
        SqlEnum(ResourceStatus, name="resource_status"),
        nullable=False,
    )

    station: Mapped["FireStationDB"] = relationship(back_populates="resources")
