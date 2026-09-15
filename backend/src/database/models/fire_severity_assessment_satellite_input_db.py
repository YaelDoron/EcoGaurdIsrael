"""SQLAlchemy ORM model linking fire-severity assessments to satellite inputs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
    from src.database.models.satellite_hotspot_db import SatelliteHotspotDB


class FireSeverityAssessmentSatelliteInputDB(Base):
    """Trace row for a satellite hotspot considered by a severity assessment."""

    __tablename__ = "fire_severity_assessment_satellite_inputs"
    __table_args__ = (
        UniqueConstraint(
            "assessment_id",
            "satellite_hotspot_id",
            name="uq_fire_severity_satellite_input_assessment_hotspot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assessment_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_severity_assessments.id"),
        nullable=False,
        index=True,
    )
    satellite_hotspot_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("satellite_hotspots.id"),
        nullable=False,
        index=True,
    )
    selected_for_frp: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    assessment: Mapped["FireSeverityAssessmentDB"] = relationship(back_populates="satellite_inputs")
    satellite_hotspot: Mapped["SatelliteHotspotDB"] = relationship()
