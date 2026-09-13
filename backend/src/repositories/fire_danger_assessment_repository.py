"""Persistence layer for fire-danger assessments and weather traceability."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.repositories.exceptions import FireDangerAssessmentRepositoryError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredFireDangerAssessment:
    """Persisted assessment plus database id and source weather traceability."""

    assessment_id: int
    assessment: FireDangerAssessment
    observation_ids: tuple[int, ...]
    station_ids: tuple[int, ...]


class FireDangerAssessmentRepository:
    """Persists FireDangerAssessment domain objects via SQLAlchemy."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Run a block of work in a session, committing on success and rolling back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_assessment(
        self,
        assessment: FireDangerAssessment,
        observation_ids: tuple[int, ...],
        station_ids: tuple[int, ...],
    ) -> StoredFireDangerAssessment:
        """Atomically store an assessment and its source weather observation trace rows."""
        self._validate_traceability(assessment, observation_ids, station_ids)

        with self._session_scope() as session:
            db_assessment = FireDangerAssessmentDB(
                area_id=assessment.area_id,
                area_name=assessment.area_name,
                area_latitude=assessment.area_latitude,
                area_longitude=assessment.area_longitude,
                area_radius_km=assessment.area_radius_km,
                assessed_at=assessment.assessed_at,
                status=assessment.status.value,
                score=assessment.score,
                danger_level=assessment.level.value if assessment.level is not None else None,
                methodology=assessment.methodology,
                methodology_version=assessment.methodology_version,
            )
            session.add(db_assessment)

            try:
                session.flush()
                for observation_id, station_id in zip(observation_ids, station_ids):
                    session.add(
                        FireDangerAssessmentWeatherInputDB(
                            assessment_id=db_assessment.id,
                            weather_observation_id=observation_id,
                            station_id=station_id,
                        )
                    )
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireDangerAssessmentRepositoryError(
                    "Fire-danger assessment persistence failed."
                ) from exc

            logger.info("Stored fire-danger assessment %s for area %s", db_assessment.id, assessment.area_id)
            return self._to_stored_assessment(db_assessment)

    def get_by_id(self, assessment_id: int) -> StoredFireDangerAssessment | None:
        """Return a persisted assessment by database id, or None if not found."""
        self._validate_assessment_id(assessment_id)
        with self._session_scope() as session:
            db_assessment = (
                session.execute(
                    select(FireDangerAssessmentDB)
                    .options(selectinload(FireDangerAssessmentDB.weather_inputs))
                    .where(FireDangerAssessmentDB.id == assessment_id)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_assessment(db_assessment) if db_assessment is not None else None

    def get_latest_for_area(self, area_id: str) -> StoredFireDangerAssessment | None:
        """Return the latest persisted assessment for an area, newest assessed_at first."""
        if not isinstance(area_id, str) or not area_id.strip():
            raise FireDangerAssessmentRepositoryError(f"area_id must be a non-empty string, got {area_id!r}")
        with self._session_scope() as session:
            db_assessment = (
                session.execute(
                    select(FireDangerAssessmentDB)
                    .options(selectinload(FireDangerAssessmentDB.weather_inputs))
                    .where(FireDangerAssessmentDB.area_id == area_id)
                    .order_by(FireDangerAssessmentDB.assessed_at.desc(), FireDangerAssessmentDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_assessment(db_assessment) if db_assessment is not None else None

    @staticmethod
    def _validate_traceability(
        assessment: FireDangerAssessment,
        observation_ids: tuple[int, ...],
        station_ids: tuple[int, ...],
    ) -> None:
        if not isinstance(assessment, FireDangerAssessment):
            raise FireDangerAssessmentRepositoryError(
                f"assessment must be a FireDangerAssessment, got {assessment!r}"
            )
        observation_ids = tuple(observation_ids)
        station_ids = tuple(station_ids)
        if len(observation_ids) != len(station_ids):
            raise FireDangerAssessmentRepositoryError(
                "observation_ids and station_ids must have the same length."
            )
        for field_name, values in (("observation_ids", observation_ids), ("station_ids", station_ids)):
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise FireDangerAssessmentRepositoryError(
                        f"{field_name} must contain positive integer ids, got {value!r}."
                    )
        if assessment.status is FireDangerAssessmentStatus.VALID and not observation_ids:
            raise FireDangerAssessmentRepositoryError(
                "VALID fire-danger assessments require at least one source observation."
            )

    @staticmethod
    def _validate_assessment_id(assessment_id: int) -> None:
        if isinstance(assessment_id, bool) or not isinstance(assessment_id, int) or assessment_id <= 0:
            raise FireDangerAssessmentRepositoryError(
                f"assessment_id must be a positive integer, got {assessment_id!r}."
            )

    @staticmethod
    def _to_stored_assessment(db_assessment: FireDangerAssessmentDB) -> StoredFireDangerAssessment:
        weather_inputs = sorted(db_assessment.weather_inputs, key=lambda trace: trace.id)
        return StoredFireDangerAssessment(
            assessment_id=db_assessment.id,
            assessment=FireDangerAssessment(
                area_id=db_assessment.area_id,
                area_name=db_assessment.area_name,
                area_latitude=db_assessment.area_latitude,
                area_longitude=db_assessment.area_longitude,
                area_radius_km=db_assessment.area_radius_km,
                assessed_at=FireDangerAssessmentRepository._ensure_aware_datetime(
                    db_assessment.assessed_at
                ),
                status=FireDangerAssessmentStatus(db_assessment.status),
                score=db_assessment.score,
                level=(
                    FireDangerLevel(db_assessment.danger_level)
                    if db_assessment.danger_level is not None
                    else None
                ),
                methodology=db_assessment.methodology,
                methodology_version=db_assessment.methodology_version,
            ),
            observation_ids=tuple(trace.weather_observation_id for trace in weather_inputs),
            station_ids=tuple(trace.station_id for trace in weather_inputs),
        )

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
