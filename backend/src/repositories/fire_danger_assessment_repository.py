"""Persistence layer for fire-danger assessments and weather traceability."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
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
    """Persisted assessment plus database id and source weather traceability.

    `created_at` is the row's own DB-insert timestamp (server-assigned, set
    once via the ORM column default, never updated) - distinct from
    `assessment.assessed_at` (the business "as-of" instant the assessment
    applies to). Exposed for the Activity Feed's Weather Conditions signal,
    which represents "when this weather summary became available to
    EcoGuard" rather than a source observation time.
    """

    assessment_id: int
    assessment: FireDangerAssessment
    observation_ids: tuple[int, ...]
    station_ids: tuple[int, ...]
    created_at: datetime


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

    def get_recent(self, limit: int) -> tuple[StoredFireDangerAssessment, ...]:
        """Return the `limit` most recently assessed rows across ALL areas, newest first.

        Task A6: distinct from get_latest_for_all_areas (latest PER area) -
        this is a bounded "recent across every assessment" read for the
        Activity Feed. Same deterministic ordering as get_latest_for_area
        (assessed_at desc, id desc). Does not populate observation_ids/
        station_ids (left empty), same rationale as get_latest_for_all_areas:
        avoids an eager-load join across `limit` rows that this batched,
        summary-only read does not need.
        """
        self._validate_limit(limit)
        with self._session_scope() as session:
            db_assessments = (
                session.execute(
                    select(FireDangerAssessmentDB)
                    .order_by(FireDangerAssessmentDB.assessed_at.desc(), FireDangerAssessmentDB.id.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return tuple(
                StoredFireDangerAssessment(
                    assessment_id=db_assessment.id,
                    assessment=FireDangerAssessment(
                        area_id=db_assessment.area_id,
                        area_name=db_assessment.area_name,
                        area_latitude=db_assessment.area_latitude,
                        area_longitude=db_assessment.area_longitude,
                        area_radius_km=db_assessment.area_radius_km,
                        assessed_at=self._ensure_aware_datetime(db_assessment.assessed_at),
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
                    observation_ids=(),
                    station_ids=(),
                    created_at=self._ensure_aware_datetime(db_assessment.created_at),
                )
                for db_assessment in db_assessments
            )

    def get_recent_with_level_in(
        self, levels: tuple[FireDangerLevel, ...], limit: int
    ) -> tuple[StoredFireDangerAssessment, ...]:
        """Return the `limit` most recent VALID assessments whose danger_level
        is one of `levels`, newest first, WITH observation_ids/station_ids
        populated (unlike get_recent).

        Added for the Activity Feed's weather-conditions signal (Part G):
        that signal needs the exact traced weather inputs for each
        qualifying assessment, not just the summary fields get_recent
        returns. Same deterministic ordering as get_recent (assessed_at
        desc, id desc). An empty `levels` tuple always returns ().
        """
        self._validate_limit(limit)
        level_values = tuple(self._validate_levels(levels))
        if not level_values:
            return ()
        with self._session_scope() as session:
            db_assessments = (
                session.execute(
                    select(FireDangerAssessmentDB)
                    .options(selectinload(FireDangerAssessmentDB.weather_inputs))
                    .where(FireDangerAssessmentDB.danger_level.in_(level_values))
                    .order_by(FireDangerAssessmentDB.assessed_at.desc(), FireDangerAssessmentDB.id.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_assessment(db_assessment) for db_assessment in db_assessments)

    def get_latest_for_all_areas(self) -> tuple[StoredFireDangerAssessment, ...]:
        """Return every area's latest persisted assessment, batched in one query.

        Task A4: an "area" here means exactly "has at least one persisted
        FireDangerAssessment row" - this repository has no independent area
        registry to enumerate against (see src/models/fire_danger_areas.py
        for the full rationale). Same "latest" ordering as
        get_latest_for_area (assessed_at desc, id desc) per area_id, resolved
        for every area at once via a window function (mirroring
        FireSeverityAssessmentRepository.get_latest_for_events) instead of
        one query per area - avoids an N+1 query pattern for the Fire Danger
        areas/latest endpoint, which the dashboard may poll every ~2 seconds.
        Does not populate observation_ids/station_ids (left empty) since this
        batched path does not join the input-trace tables; callers needing
        those should look up the returned assessment_id via get_by_id.
        Returns an empty tuple if no assessment has ever been persisted.
        Result order is not guaranteed sorted by area name - callers that
        need a deterministic display order (e.g. FireDangerQueryService)
        sort it themselves.
        """
        with self._session_scope() as session:
            row_number = (
                func.row_number()
                .over(
                    partition_by=FireDangerAssessmentDB.area_id,
                    order_by=(
                        FireDangerAssessmentDB.assessed_at.desc(),
                        FireDangerAssessmentDB.id.desc(),
                    ),
                )
                .label("row_number")
            )
            ranked = select(FireDangerAssessmentDB, row_number).subquery()
            rows = session.execute(select(ranked).where(ranked.c.row_number == 1)).all()
            return tuple(
                StoredFireDangerAssessment(
                    assessment_id=row.id,
                    assessment=FireDangerAssessment(
                        area_id=row.area_id,
                        area_name=row.area_name,
                        area_latitude=row.area_latitude,
                        area_longitude=row.area_longitude,
                        area_radius_km=row.area_radius_km,
                        assessed_at=self._ensure_aware_datetime(row.assessed_at),
                        status=FireDangerAssessmentStatus(row.status),
                        score=row.score,
                        level=FireDangerLevel(row.danger_level) if row.danger_level is not None else None,
                        methodology=row.methodology,
                        methodology_version=row.methodology_version,
                    ),
                    observation_ids=(),
                    station_ids=(),
                    created_at=self._ensure_aware_datetime(row.created_at),
                )
                for row in rows
            )

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
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise FireDangerAssessmentRepositoryError(f"Invalid limit: {limit!r}. Must be a positive integer.")

    @staticmethod
    def _validate_levels(levels: tuple[FireDangerLevel, ...]) -> tuple[str, ...]:
        levels = tuple(levels)
        for level in levels:
            if not isinstance(level, FireDangerLevel):
                raise FireDangerAssessmentRepositoryError(f"levels must contain FireDangerLevel values, got {level!r}")
        return tuple(level.value for level in levels)

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
            created_at=FireDangerAssessmentRepository._ensure_aware_datetime(db_assessment.created_at),
        )

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
