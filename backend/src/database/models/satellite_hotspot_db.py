"""SQLAlchemy ORM model for the satellite_hotspots table.

This is the persistence-layer counterpart of the SatelliteHotspot domain
dataclass. A stored row is still only a satellite thermal anomaly detection,
not a confirmed wildfire.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class SatelliteHotspotDB(Base):
    """A persisted satellite thermal anomaly / active-fire detection."""

    __tablename__ = "satellite_hotspots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_key: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    confidence: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brightness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    satellite: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    instrument: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    day_night: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    location_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # When EcoGuard itself persisted this hotspot (Task 4's Activity Feed
    # "available_at"), NOT derived from detected_at. Nullable because
    # pre-migration rows genuinely have no such recorded moment - never
    # backfilled with a fabricated value. Set once, on insert, never updated.
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=lambda: datetime.now(timezone.utc)
    )
