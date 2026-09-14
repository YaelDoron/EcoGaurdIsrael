"""SQLAlchemy ORM model linking fire events to satellite hotspot evidence."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_event_db import FireEventDB
    from src.database.models.satellite_hotspot_db import SatelliteHotspotDB


class FireEventSatelliteEvidenceDB(Base):
    """Relationship from a FireEvent to a persisted SatelliteHotspot row."""

    __tablename__ = "fire_event_satellite_evidence"
    __table_args__ = (
        UniqueConstraint(
            "fire_event_id",
            "satellite_hotspot_id",
            name="uq_fire_event_satellite_evidence_event_hotspot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    satellite_hotspot_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("satellite_hotspots.id"),
        nullable=False,
        index=True,
    )

    fire_event: Mapped["FireEventDB"] = relationship(back_populates="satellite_evidence")
    satellite_hotspot: Mapped["SatelliteHotspotDB"] = relationship()
