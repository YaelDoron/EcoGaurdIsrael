"""Persistence layer for fire-severity assessments and traceability."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_severity_assessment_satellite_input_db import (
    FireSeverityAssessmentSatelliteInputDB,
)
from src.database.models.fire_severity_assessment_weather_input_db import (
    FireSeverityAssessmentWeatherInputDB,
)
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.repositories.exceptions import FireSeverityAssessmentRepositoryError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredFireSeverityAssessment:
    """Persisted severity assessment plus database identity and input traces."""

    assessment_id: int
    assessment: FireSeverityAssessment
    weather_observation_ids: tuple[int, ...] = ()
    satellite_hotspot_ids: tuple[int, ...] = ()
    selected_frp_hotspot_id: int | None = None


class FireSeverityAssessmentRepository:
    """Persists FireSeverityAssessment domain objects via SQLAlchemy."""

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
        assessment: FireSeverityAssessment,
        weather_observation_ids: tuple[int, ...],
        satellite_hotspot_ids: tuple[int, ...],
        selected_frp_hotspot_id: int | None,
    ) -> StoredFireSeverityAssessment:
        """Atomically store an assessment and its input trace rows."""
        weather_ids = self._normalize_ids("weather_observation_ids", weather_observation_ids)
        satellite_ids = self._normalize_ids("satellite_hotspot_ids", satellite_hotspot_ids)
        self._validate_save_request(assessment, weather_ids, satellite_ids, selected_frp_hotspot_id)

        with self._session_scope() as session:
            db_assessment = self._to_db_assessment(assessment)
            session.add(db_assessment)
            try:
                session.flush()
                for weather_id in weather_ids:
                    session.add(
                        FireSeverityAssessmentWeatherInputDB(
                            assessment_id=db_assessment.id,
                            weather_observation_id=weather_id,
                        )
                    )
                for satellite_id in satellite_ids:
                    session.add(
                        FireSeverityAssessmentSatelliteInputDB(
                            assessment_id=db_assessment.id,
                            satellite_hotspot_id=satellite_id,
                            selected_for_frp=satellite_id == selected_frp_hotspot_id,
                        )
                    )
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireSeverityAssessmentRepositoryError(
                    "Fire-severity assessment persistence failed."
                ) from exc

            logger.info(
                "Stored fire-severity assessment %s for FireEvent %s",
                db_assessment.id,
                assessment.fire_event_id,
            )
            return self._to_stored_assessment(db_assessment)

    def get_by_id(self, assessment_id: int) -> StoredFireSeverityAssessment | None:
        """Return a persisted severity assessment by database id, or None if absent."""
        self._validate_assessment_id(assessment_id)
        with self._session_scope() as session:
            db_assessment = self._get_db_assessment(session, assessment_id)
            return self._to_stored_assessment(db_assessment) if db_assessment is not None else None

    def get_latest_for_event(self, fire_event_id: int) -> StoredFireSeverityAssessment | None:
        """Return the latest persisted severity assessment for a FireEvent."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            db_assessment = (
                session.execute(
                    select(FireSeverityAssessmentDB)
                    .options(
                        selectinload(FireSeverityAssessmentDB.weather_inputs),
                        selectinload(FireSeverityAssessmentDB.satellite_inputs),
                    )
                    .where(FireSeverityAssessmentDB.fire_event_id == fire_event_id)
                    .order_by(FireSeverityAssessmentDB.assessed_at.desc(), FireSeverityAssessmentDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_assessment(db_assessment) if db_assessment is not None else None

    def get_latest_for_event_as_of(
        self,
        fire_event_id: int,
        as_of: datetime,
    ) -> StoredFireSeverityAssessment | None:
        """Return the latest persisted severity state for a FireEvent at or before `as_of`."""
        self._validate_fire_event_id(fire_event_id)
        self._validate_aware_datetime("as_of", as_of)
        with self._session_scope() as session:
            db_assessment = (
                session.execute(
                    select(FireSeverityAssessmentDB)
                    .options(
                        selectinload(FireSeverityAssessmentDB.weather_inputs),
                        selectinload(FireSeverityAssessmentDB.satellite_inputs),
                    )
                    .where(
                        FireSeverityAssessmentDB.fire_event_id == fire_event_id,
                        FireSeverityAssessmentDB.assessed_at <= as_of,
                    )
                    .order_by(FireSeverityAssessmentDB.assessed_at.desc(), FireSeverityAssessmentDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_assessment(db_assessment) if db_assessment is not None else None

    def get_latest_for_events(self, fire_event_ids: Iterable[int]) -> dict[int, StoredFireSeverityAssessment]:
        """Return each event's latest persisted severity assessment, batched in one query.

        Same "latest" ordering as get_latest_for_event (assessed_at desc, id
        desc) per fire_event_id, resolved for many events at once via a
        window function instead of one query per event - avoids N+1 queries
        when listing many active FireEvents (US 6.1). Does not populate
        weather_observation_ids/satellite_hotspot_ids/selected_frp_hotspot_id
        (left at their StoredFireSeverityAssessment defaults) since this
        batched path does not join the input-trace tables; callers needing
        those should look up the returned assessment_id via get_by_id.
        FireEvent ids with no persisted assessment are simply absent from
        the returned mapping.
        """
        ids = self._normalize_ids("fire_event_ids", tuple(fire_event_ids))
        if not ids:
            return {}
        with self._session_scope() as session:
            row_number = (
                func.row_number()
                .over(
                    partition_by=FireSeverityAssessmentDB.fire_event_id,
                    order_by=(
                        FireSeverityAssessmentDB.assessed_at.desc(),
                        FireSeverityAssessmentDB.id.desc(),
                    ),
                )
                .label("row_number")
            )
            ranked = (
                select(FireSeverityAssessmentDB, row_number)
                .where(FireSeverityAssessmentDB.fire_event_id.in_(ids))
                .subquery()
            )
            rows = session.execute(select(ranked).where(ranked.c.row_number == 1)).all()
            return {
                row.fire_event_id: StoredFireSeverityAssessment(
                    assessment_id=row.id,
                    assessment=FireSeverityAssessment(
                        fire_event_id=row.fire_event_id,
                        assessed_at=self._ensure_aware_datetime(row.assessed_at),
                        status=FireSeverityAssessmentStatus(row.status),
                        score=row.score,
                        level=FireSeverityLevel(row.severity_level) if row.severity_level is not None else None,
                        methodology=row.methodology,
                        methodology_version=row.methodology_version,
                        vegetation_source=row.vegetation_source,
                        vegetation_dataset_year=row.vegetation_dataset_year,
                        vegetation_radius_km=row.vegetation_radius_km,
                        vegetation_dominant_land_cover=row.vegetation_dominant_land_cover,
                        vegetation_fuel_score=row.vegetation_fuel_score,
                    ),
                )
                for row in rows
            }

    def get_weather_input_ids(self, assessment_id: int) -> tuple[int, ...]:
        """Return weather observation IDs linked to an assessment, sorted ascending."""
        self._validate_assessment_id(assessment_id)
        with self._session_scope() as session:
            return tuple(
                sorted(
                    session.execute(
                        select(FireSeverityAssessmentWeatherInputDB.weather_observation_id).where(
                            FireSeverityAssessmentWeatherInputDB.assessment_id == assessment_id
                        )
                    ).scalars()
                )
            )

    def get_satellite_input_ids(self, assessment_id: int) -> tuple[int, ...]:
        """Return satellite hotspot IDs linked to an assessment, sorted ascending."""
        self._validate_assessment_id(assessment_id)
        with self._session_scope() as session:
            return tuple(
                sorted(
                    session.execute(
                        select(FireSeverityAssessmentSatelliteInputDB.satellite_hotspot_id).where(
                            FireSeverityAssessmentSatelliteInputDB.assessment_id == assessment_id
                        )
                    ).scalars()
                )
            )

    def get_selected_frp_hotspot_id(self, assessment_id: int) -> int | None:
        """Return the satellite hotspot selected as the FRP source, if any."""
        self._validate_assessment_id(assessment_id)
        with self._session_scope() as session:
            return session.execute(
                select(FireSeverityAssessmentSatelliteInputDB.satellite_hotspot_id)
                .where(
                    FireSeverityAssessmentSatelliteInputDB.assessment_id == assessment_id,
                    FireSeverityAssessmentSatelliteInputDB.selected_for_frp.is_(True),
                )
                .order_by(FireSeverityAssessmentSatelliteInputDB.satellite_hotspot_id.asc())
                .limit(1)
            ).scalar_one_or_none()

    @staticmethod
    def _to_db_assessment(assessment: FireSeverityAssessment) -> FireSeverityAssessmentDB:
        return FireSeverityAssessmentDB(
            fire_event_id=assessment.fire_event_id,
            assessed_at=assessment.assessed_at,
            status=assessment.status.value,
            score=assessment.score,
            severity_level=assessment.level.value if assessment.level is not None else None,
            methodology=assessment.methodology,
            methodology_version=assessment.methodology_version,
            vegetation_source=assessment.vegetation_source,
            vegetation_dataset_year=assessment.vegetation_dataset_year,
            vegetation_radius_km=assessment.vegetation_radius_km,
            vegetation_dominant_land_cover=assessment.vegetation_dominant_land_cover,
            vegetation_fuel_score=assessment.vegetation_fuel_score,
        )

    @staticmethod
    def _get_db_assessment(session: Session, assessment_id: int) -> FireSeverityAssessmentDB | None:
        return (
            session.execute(
                select(FireSeverityAssessmentDB)
                .options(
                    selectinload(FireSeverityAssessmentDB.weather_inputs),
                    selectinload(FireSeverityAssessmentDB.satellite_inputs),
                )
                .where(FireSeverityAssessmentDB.id == assessment_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _to_stored_assessment(cls, db_assessment: FireSeverityAssessmentDB) -> StoredFireSeverityAssessment:
        weather_ids = tuple(sorted(trace.weather_observation_id for trace in db_assessment.weather_inputs))
        satellite_inputs = sorted(db_assessment.satellite_inputs, key=lambda trace: trace.satellite_hotspot_id)
        selected_ids = tuple(trace.satellite_hotspot_id for trace in satellite_inputs if trace.selected_for_frp)
        return StoredFireSeverityAssessment(
            assessment_id=db_assessment.id,
            assessment=FireSeverityAssessment(
                fire_event_id=db_assessment.fire_event_id,
                assessed_at=cls._ensure_aware_datetime(db_assessment.assessed_at),
                status=FireSeverityAssessmentStatus(db_assessment.status),
                score=db_assessment.score,
                level=(
                    FireSeverityLevel(db_assessment.severity_level)
                    if db_assessment.severity_level is not None
                    else None
                ),
                methodology=db_assessment.methodology,
                methodology_version=db_assessment.methodology_version,
                vegetation_source=db_assessment.vegetation_source,
                vegetation_dataset_year=db_assessment.vegetation_dataset_year,
                vegetation_radius_km=db_assessment.vegetation_radius_km,
                vegetation_dominant_land_cover=db_assessment.vegetation_dominant_land_cover,
                vegetation_fuel_score=db_assessment.vegetation_fuel_score,
            ),
            weather_observation_ids=weather_ids,
            satellite_hotspot_ids=tuple(trace.satellite_hotspot_id for trace in satellite_inputs),
            selected_frp_hotspot_id=selected_ids[0] if selected_ids else None,
        )

    @staticmethod
    def _validate_save_request(
        assessment: FireSeverityAssessment,
        weather_ids: tuple[int, ...],
        satellite_ids: tuple[int, ...],
        selected_frp_hotspot_id: int | None,
    ) -> None:
        if not isinstance(assessment, FireSeverityAssessment):
            raise FireSeverityAssessmentRepositoryError(
                f"assessment must be a FireSeverityAssessment, got {assessment!r}"
            )
        if selected_frp_hotspot_id is not None and (
            isinstance(selected_frp_hotspot_id, bool)
            or not isinstance(selected_frp_hotspot_id, int)
            or selected_frp_hotspot_id <= 0
        ):
            raise FireSeverityAssessmentRepositoryError(
                "selected_frp_hotspot_id must be a positive integer or None, "
                f"got {selected_frp_hotspot_id!r}."
            )
        if assessment.status is FireSeverityAssessmentStatus.VALID:
            if not weather_ids:
                raise FireSeverityAssessmentRepositoryError(
                    "VALID fire-severity assessments require at least one weather input."
                )
            if not satellite_ids:
                raise FireSeverityAssessmentRepositoryError(
                    "VALID fire-severity assessments require at least one satellite input."
                )
            if selected_frp_hotspot_id is None:
                raise FireSeverityAssessmentRepositoryError(
                    "VALID fire-severity assessments require selected_frp_hotspot_id."
                )
            if selected_frp_hotspot_id not in satellite_ids:
                raise FireSeverityAssessmentRepositoryError(
                    "selected_frp_hotspot_id must be included in satellite_hotspot_ids."
                )
        elif selected_frp_hotspot_id is not None and selected_frp_hotspot_id not in satellite_ids:
            raise FireSeverityAssessmentRepositoryError(
                "selected_frp_hotspot_id must be included in satellite_hotspot_ids when provided."
            )

    @staticmethod
    def _normalize_ids(field_name: str, ids: tuple[int, ...]) -> tuple[int, ...]:
        try:
            values = tuple(ids)
        except TypeError as exc:
            raise FireSeverityAssessmentRepositoryError(f"{field_name} must be iterable.") from exc
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FireSeverityAssessmentRepositoryError(
                    f"{field_name} must contain positive integer ids, got {value!r}."
                )
        return tuple(sorted(set(values)))

    @staticmethod
    def _validate_assessment_id(assessment_id: int) -> None:
        if isinstance(assessment_id, bool) or not isinstance(assessment_id, int) or assessment_id <= 0:
            raise FireSeverityAssessmentRepositoryError(
                f"assessment_id must be a positive integer, got {assessment_id!r}."
            )

    @staticmethod
    def _validate_fire_event_id(fire_event_id: int) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise FireSeverityAssessmentRepositoryError(
                f"fire_event_id must be a positive integer, got {fire_event_id!r}."
            )

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise FireSeverityAssessmentRepositoryError(
                f"{field_name} must be a timezone-aware datetime, got {value!r}."
            )

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
