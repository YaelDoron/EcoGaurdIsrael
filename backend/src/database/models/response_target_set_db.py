"""SQLAlchemy ORM model for persisted response-target snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_target_db import ResponseTargetDB


class ResponseTargetSetDB(Base):
    """Append-only generated response-target set for one FireEvent."""

    __tablename__ = "response_target_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    methodology: Mapped[str] = mapped_column(String, nullable=False)
    methodology_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    targets: Mapped[list["ResponseTargetDB"]] = relationship(
        back_populates="target_set",
        cascade="all, delete-orphan",
    )
