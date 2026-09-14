"""Persistence layer for wildfire events and their evidence traceability."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.exceptions import FireEventRepositoryError
from src.repositories.fire_event_config import (
    ACTIVE_EVENT_MATCH_DISTANCE_KM,
    ACTIVE_EVENT_MATCH_WINDOW_HOURS,
)

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0088
_DISTANCE_TOLERANCE_KM = 1e-9
_ACTIVE_STATUSES = {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}


@dataclass(frozen=True)
class StoredFireEvent:
    """Persisted FireEvent plus database identity and source-aware evidence refs."""

    id: int
    event: FireEvent
    supporting_evidence: tuple[FireEvidenceRef, ...] = ()


class FireEventRepository:
    """Persists FireEvent domain objects and evidence associations."""

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

    def create_event(
        self,
        event: FireEvent,
        supporting_evidence: tuple[FireEvidenceRef, ...],
    ) -> StoredFireEvent:
        """Atomically persist a new active event and its direct evidence refs."""
        self._validate_event(event)
        if event.status not in _ACTIVE_STATUSES:
            raise FireEventRepositoryError("Detection-created FireEvents must be SUSPECTED or CONFIRMED.")
        evidence_refs = self._normalize_evidence_refs(supporting_evidence)
        if not evidence_refs:
            raise FireEventRepositoryError("FireEvent creation requires at least one supporting evidence ref.")

        with self._session_scope() as session:
            db_event = self._to_db_event(event)
            session.add(db_event)
            try:
                session.flush()
                self._insert_missing_evidence_refs(session, db_event.id, evidence_refs)
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireEventRepositoryError("FireEvent persistence failed.") from exc

            logger.info("Stored FireEvent %s with %s evidence refs", db_event.id, len(evidence_refs))
            return self._to_stored_event(db_event, evidence_refs)

    def get_by_id(self, fire_event_id: int) -> StoredFireEvent | None:
        """Return a persisted FireEvent by id, or None when no row exists."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            db_event = self._get_db_event(session, fire_event_id)
            if db_event is None:
                return None
            return self._to_stored_event(db_event, self._get_evidence_refs(session, fire_event_id))

    def get_evidence_refs(self, fire_event_id: int) -> tuple[FireEvidenceRef, ...]:
        """Return source-aware evidence refs for a FireEvent in deterministic order."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            return self._get_evidence_refs(session, fire_event_id)

    def attach_evidence(
        self,
        fire_event_id: int,
        evidence: tuple[FireEvidenceRef, ...],
    ) -> tuple[FireEvidenceRef, ...]:
        """Attach new evidence refs idempotently and return all stored refs."""
        self._validate_fire_event_id(fire_event_id)
        evidence_refs = self._normalize_evidence_refs(evidence)

        with self._session_scope() as session:
            if self._get_db_event(session, fire_event_id) is None:
                raise FireEventRepositoryError(f"FireEvent {fire_event_id!r} was not found.")
            try:
                self._insert_missing_evidence_refs(session, fire_event_id, evidence_refs)
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise FireEventRepositoryError("FireEvent evidence attachment failed.") from exc
            return self._get_evidence_refs(session, fire_event_id)

    def update_event(self, fire_event_id: int, event: FireEvent) -> StoredFireEvent:
        """Update caller-supplied event fields without recalculating confidence or status."""
        self._validate_fire_event_id(fire_event_id)
        self._validate_event(event)

        with self._session_scope() as session:
            db_event = self._get_db_event(session, fire_event_id)
            if db_event is None:
                raise FireEventRepositoryError(f"FireEvent {fire_event_id!r} was not found.")

            db_event.latitude = event.latitude
            db_event.longitude = event.longitude
            db_event.updated_at = event.updated_at
            db_event.status = event.status.value
            db_event.detection_confidence = event.detection_confidence
            try:
                session.flush()
            except SQLAlchemyError as exc:
                raise FireEventRepositoryError("FireEvent update failed.") from exc

            return self._to_stored_event(db_event, self._get_evidence_refs(session, fire_event_id))

    def find_matching_active_event(
        self,
        latitude: float,
        longitude: float,
        observed_at: datetime,
    ) -> StoredFireEvent | None:
        """Find a nearby active event using updated_at as the event recency timestamp.

        Eligible matches must be active, within ACTIVE_EVENT_MATCH_DISTANCE_KM,
        and within ACTIVE_EVENT_MATCH_WINDOW_HOURS of updated_at. Multiple
        matches are ordered by distance, time difference, most recent update,
        then smallest event id.
        """
        self._validate_coordinate("latitude", latitude, -90, 90)
        self._validate_coordinate("longitude", longitude, -180, 180)
        self._validate_aware_datetime("observed_at", observed_at)
        window = timedelta(hours=ACTIVE_EVENT_MATCH_WINDOW_HOURS)
        start_time = observed_at - window
        end_time = observed_at + window

        with self._session_scope() as session:
            db_events = (
                session.execute(
                    select(FireEventDB)
                    .options(
                        selectinload(FireEventDB.satellite_evidence),
                        selectinload(FireEventDB.news_evidence),
                    )
                    .where(
                        FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES),
                        FireEventDB.updated_at >= start_time,
                        FireEventDB.updated_at <= end_time,
                    )
                )
                .scalars()
                .all()
            )

            matches = []
            for db_event in db_events:
                distance_km = self._haversine_distance_km(
                    latitude,
                    longitude,
                    db_event.latitude,
                    db_event.longitude,
                )
                if distance_km > ACTIVE_EVENT_MATCH_DISTANCE_KM + _DISTANCE_TOLERANCE_KM:
                    continue
                event_updated_at = self._ensure_aware_datetime(db_event.updated_at)
                time_difference_seconds = abs((observed_at - event_updated_at).total_seconds())
                matches.append(
                    (
                        distance_km,
                        time_difference_seconds,
                        -event_updated_at.timestamp(),
                        db_event.id,
                        db_event,
                    )
                )

            if not matches:
                return None
            db_event = sorted(matches, key=lambda match: match[:4])[0][4]
            return self._to_stored_event(db_event, self._get_evidence_refs(session, db_event.id))

    def _insert_missing_evidence_refs(
        self,
        session: Session,
        fire_event_id: int,
        evidence_refs: tuple[FireEvidenceRef, ...],
    ) -> None:
        existing_refs = set(self._get_evidence_refs(session, fire_event_id))
        for evidence_ref in evidence_refs:
            if evidence_ref in existing_refs:
                continue
            if evidence_ref.evidence_type is FireEvidenceType.SATELLITE:
                session.add(
                    FireEventSatelliteEvidenceDB(
                        fire_event_id=fire_event_id,
                        satellite_hotspot_id=evidence_ref.evidence_id,
                    )
                )
            elif evidence_ref.evidence_type is FireEvidenceType.NEWS:
                session.add(
                    FireEventNewsEvidenceDB(
                        fire_event_id=fire_event_id,
                        wildfire_report_id=evidence_ref.evidence_id,
                    )
                )
            else:
                raise FireEventRepositoryError(f"Unsupported evidence type: {evidence_ref.evidence_type!r}.")
            existing_refs.add(evidence_ref)

    @staticmethod
    def _to_db_event(event: FireEvent) -> FireEventDB:
        return FireEventDB(
            latitude=event.latitude,
            longitude=event.longitude,
            detected_at=event.detected_at,
            updated_at=event.updated_at,
            status=event.status.value,
            detection_confidence=event.detection_confidence,
            methodology=event.methodology,
            methodology_version=event.methodology_version,
        )

    @classmethod
    def _to_stored_event(
        cls,
        db_event: FireEventDB,
        evidence_refs: tuple[FireEvidenceRef, ...] = (),
    ) -> StoredFireEvent:
        return StoredFireEvent(
            id=db_event.id,
            event=FireEvent(
                latitude=db_event.latitude,
                longitude=db_event.longitude,
                detected_at=cls._ensure_aware_datetime(db_event.detected_at),
                updated_at=cls._ensure_aware_datetime(db_event.updated_at),
                status=FireEventStatus(db_event.status),
                detection_confidence=db_event.detection_confidence,
                methodology=db_event.methodology,
                methodology_version=db_event.methodology_version,
            ),
            supporting_evidence=evidence_refs,
        )

    @staticmethod
    def _get_db_event(session: Session, fire_event_id: int) -> FireEventDB | None:
        return (
            session.execute(
                select(FireEventDB)
                .options(
                    selectinload(FireEventDB.satellite_evidence),
                    selectinload(FireEventDB.news_evidence),
                )
                .where(FireEventDB.id == fire_event_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _get_evidence_refs(cls, session: Session, fire_event_id: int) -> tuple[FireEvidenceRef, ...]:
        satellite_refs = [
            FireEvidenceRef(FireEvidenceType.SATELLITE, satellite_hotspot_id)
            for satellite_hotspot_id in session.execute(
                select(FireEventSatelliteEvidenceDB.satellite_hotspot_id).where(
                    FireEventSatelliteEvidenceDB.fire_event_id == fire_event_id
                )
            ).scalars()
        ]
        news_refs = [
            FireEvidenceRef(FireEvidenceType.NEWS, wildfire_report_id)
            for wildfire_report_id in session.execute(
                select(FireEventNewsEvidenceDB.wildfire_report_id).where(
                    FireEventNewsEvidenceDB.fire_event_id == fire_event_id
                )
            ).scalars()
        ]
        return cls._sort_evidence_refs(tuple(satellite_refs + news_refs))

    @staticmethod
    def _normalize_evidence_refs(evidence_refs: Iterable[FireEvidenceRef]) -> tuple[FireEvidenceRef, ...]:
        try:
            refs = tuple(evidence_refs)
        except TypeError as exc:
            raise FireEventRepositoryError("evidence refs must be iterable.") from exc
        for evidence_ref in refs:
            if not isinstance(evidence_ref, FireEvidenceRef):
                raise FireEventRepositoryError(f"evidence must contain FireEvidenceRef items, got {evidence_ref!r}.")
        if len(set(refs)) != len(refs):
            raise FireEventRepositoryError("evidence refs must not contain duplicates.")
        return FireEventRepository._sort_evidence_refs(refs)

    @staticmethod
    def _sort_evidence_refs(evidence_refs: tuple[FireEvidenceRef, ...]) -> tuple[FireEvidenceRef, ...]:
        return tuple(sorted(evidence_refs, key=lambda ref: (ref.evidence_type.value, ref.evidence_id)))

    @staticmethod
    def _validate_event(event: FireEvent) -> None:
        if not isinstance(event, FireEvent):
            raise FireEventRepositoryError(f"event must be a FireEvent, got {event!r}.")

    @staticmethod
    def _validate_fire_event_id(fire_event_id: int) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise FireEventRepositoryError(f"fire_event_id must be a positive integer, got {fire_event_id!r}.")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise FireEventRepositoryError(f"{field_name} must be a timezone-aware datetime, got {value!r}.")

    @staticmethod
    def _validate_coordinate(field_name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise FireEventRepositoryError(f"{field_name} must be a finite number, got {value!r}.")
        if not minimum <= value <= maximum:
            raise FireEventRepositoryError(f"{field_name} must be within [{minimum}, {maximum}], got {value!r}.")

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _haversine_distance_km(
        first_latitude: float,
        first_longitude: float,
        second_latitude: float,
        second_longitude: float,
    ) -> float:
        first_latitude_rad = math.radians(first_latitude)
        second_latitude_rad = math.radians(second_latitude)
        latitude_delta = math.radians(second_latitude - first_latitude)
        longitude_delta = math.radians(second_longitude - first_longitude)
        a = (
            math.sin(latitude_delta / 2) ** 2
            + math.cos(first_latitude_rad)
            * math.cos(second_latitude_rad)
            * math.sin(longitude_delta / 2) ** 2
        )
        return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
