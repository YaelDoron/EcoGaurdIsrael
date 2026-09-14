"""SQLAlchemy ORM model linking fire events to wildfire news evidence."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_event_db import FireEventDB
    from src.database.models.wildfire_report_db import WildfireReportDB


class FireEventNewsEvidenceDB(Base):
    """Relationship from a FireEvent to a persisted WildfireReport row."""

    __tablename__ = "fire_event_news_evidence"
    __table_args__ = (
        UniqueConstraint(
            "fire_event_id",
            "wildfire_report_id",
            name="uq_fire_event_news_evidence_event_report",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    wildfire_report_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("wildfire_reports.id"),
        nullable=False,
        index=True,
    )

    fire_event: Mapped["FireEventDB"] = relationship(back_populates="news_evidence")
    wildfire_report: Mapped["WildfireReportDB"] = relationship()
