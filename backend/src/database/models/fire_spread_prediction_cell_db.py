"""SQLAlchemy ORM model for one predicted cell of a wildfire-spread prediction run."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB


class FireSpreadPredictionCellDB(Base):
    """A single predicted grid cell produced by a wildfire-spread prediction run.

    Row/column CA grid indices are intentionally not persisted -- the domain
    model (FireSpreadPredictionCell) does not carry them either, only the
    resolved geographic coordinate.
    """

    __tablename__ = "fire_spread_prediction_cells"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_spread_predictions.id"),
        nullable=False,
        index=True,
    )
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    spread_probability: Mapped[float] = mapped_column(Float, nullable=False)
    spread_risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    reached_step: Mapped[int] = mapped_column(Integer, nullable=False)
    reached_minutes: Mapped[int] = mapped_column(Integer, nullable=False)

    prediction: Mapped["FireSpreadPredictionDB"] = relationship(back_populates="cells")
