"""SQLAlchemy ORM model for wildfire news reports."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class WildfireReportDB(Base):
    """A persisted wildfire news report discovered from RSS/news sources."""

    __tablename__ = "wildfire_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_url: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    source_feed: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    location_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # NewsWildfireSignalStrength.value ("none"/"weak"/"moderate"/"strong") or
    # NULL. NULL means unknown/unavailable (including every row saved before
    # this column existed) - never backfilled, never fabricated as "none".
    wildfire_signal_strength: Mapped[Optional[str]] = mapped_column(String, nullable=True)
