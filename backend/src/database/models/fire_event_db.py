"""SQLAlchemy ORM model for persisted wildfire events."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
    from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB


class FireEventDB(Base):
    """A persisted wildfire event created from direct detection evidence."""

    __tablename__ = "fire_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    location_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    satellite_evidence: Mapped[list["FireEventSatelliteEvidenceDB"]] = relationship(
        back_populates="fire_event",
        cascade="all, delete-orphan",
    )
    news_evidence: Mapped[list["FireEventNewsEvidenceDB"]] = relationship(
        back_populates="fire_event",
        cascade="all, delete-orphan",
    )
