"""SQLAlchemy ORM model for one persisted response target."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.response_target_set_db import ResponseTargetSetDB


class ResponseTargetDB(Base):
    """One ordered target inside a persisted response-target set."""

    __tablename__ = "response_targets"
    __table_args__ = (
        UniqueConstraint("response_target_set_id", "target_order", name="uq_response_targets_set_order"),
        CheckConstraint("target_order >= 0", name="ck_response_targets_target_order_non_negative"),
        CheckConstraint("latitude >= -90 AND latitude <= 90", name="ck_response_targets_latitude_range"),
        CheckConstraint("longitude >= -180 AND longitude <= 180", name="ck_response_targets_longitude_range"),
        CheckConstraint("priority_score >= 0", name="ck_response_targets_priority_non_negative"),
        CheckConstraint(
            "("
            "target_type = 'active_fire' "
            "AND prediction_horizon_minutes IS NULL "
            "AND spread_prediction_id IS NULL "
            "AND spread_prediction_cell_id IS NULL"
            ") OR ("
            "target_type = 'predicted_risk' "
            "AND prediction_horizon_minutes IS NOT NULL "
            "AND prediction_horizon_minutes > 0 "
            "AND spread_prediction_id IS NOT NULL "
            "AND spread_prediction_cell_id IS NOT NULL"
            ")",
            name="ck_response_targets_type_source_metadata",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_target_set_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_target_sets.id"),
        nullable=False,
        index=True,
    )
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    target_order: Mapped[int] = mapped_column(Integer, nullable=False)
    target_type: Mapped[str] = mapped_column(String, nullable=False)

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)

    prediction_horizon_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    spread_prediction_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("fire_spread_predictions.id"),
        nullable=True,
        index=True,
    )
    spread_prediction_cell_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("fire_spread_prediction_cells.id"),
        nullable=True,
        index=True,
    )

    target_set: Mapped["ResponseTargetSetDB"] = relationship(back_populates="targets")
