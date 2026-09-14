"""Persistence layer for satellite hotspot detections.

SatelliteHotspotRepository bridges SatelliteHotspot domain dataclasses and the
SQLAlchemy SatelliteHotspotDB model. It only handles persistence: no FIRMS API
calls, CSV parsing, geographic filtering, scheduling, or wildfire confirmation.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.exceptions import SatelliteHotspotRepositoryError

logger = logging.getLogger(__name__)

_NONE_SATELLITE_SENTINEL = "<NONE>"


@dataclass(frozen=True)
class SaveHotspotResult:
    """Result of SatelliteHotspotRepository.save_hotspot()."""

    hotspot: SatelliteHotspot
    is_duplicate: bool


@dataclass(frozen=True)
class StoredSatelliteHotspot:
    """Persisted satellite hotspot with database identity."""

    id: int
    hotspot: SatelliteHotspot


class SatelliteHotspotRepository:
    """Persists and retrieves SatelliteHotspot objects via SQLAlchemy."""

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

    def save_hotspot(self, hotspot: SatelliteHotspot) -> SaveHotspotResult:
        """Insert a hotspot unless its deterministic detection key already exists."""
        detection_key = self._generate_detection_key(hotspot)

        with self._session_scope() as session:
            existing = self._find_by_detection_key(session, detection_key)
            if existing is not None:
                logger.info("Skipped duplicate satellite hotspot at %s", hotspot.detected_at)
                return SaveHotspotResult(hotspot=self._to_domain_hotspot(existing), is_duplicate=True)

            db_hotspot = SatelliteHotspotDB(
                detection_key=detection_key,
                latitude=hotspot.latitude,
                longitude=hotspot.longitude,
                detected_at=hotspot.detected_at,
                confidence=hotspot.confidence,
                frp=hotspot.frp,
                brightness=hotspot.brightness,
                satellite=hotspot.satellite,
                instrument=hotspot.instrument,
                day_night=hotspot.day_night,
            )
            session.add(db_hotspot)

            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                logger.info("Duplicate satellite hotspot detected on write at %s", hotspot.detected_at)
                existing = self._find_by_detection_key(session, detection_key)
                if existing is None:
                    raise SatelliteHotspotRepositoryError(
                        "Satellite hotspot write failed and duplicate row could not be found."
                    ) from None
                return SaveHotspotResult(hotspot=self._to_domain_hotspot(existing), is_duplicate=True)
            except SQLAlchemyError as exc:
                raise SatelliteHotspotRepositoryError("Satellite hotspot write failed.") from exc

            logger.info("Stored satellite hotspot at %s", hotspot.detected_at)
            return SaveHotspotResult(hotspot=self._to_domain_hotspot(db_hotspot), is_duplicate=False)

    def get_latest_hotspots(self, limit: int = 100) -> list[SatelliteHotspot]:
        """Return latest hotspots newest first, limited at the SQL level."""
        self._validate_limit(limit)
        with self._session_scope() as session:
            db_hotspots = (
                session.execute(
                    select(SatelliteHotspotDB)
                    .order_by(SatelliteHotspotDB.detected_at.desc(), SatelliteHotspotDB.id.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            return [self._to_domain_hotspot(db_hotspot) for db_hotspot in db_hotspots]

    def get_hotspots_between(self, start_time: datetime, end_time: datetime) -> list[SatelliteHotspot]:
        """Return hotspots detected between two datetimes, newest first."""
        if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
            raise SatelliteHotspotRepositoryError("start_time and end_time must be datetime instances.")
        if start_time > end_time:
            raise SatelliteHotspotRepositoryError("start_time must be less than or equal to end_time.")

        with self._session_scope() as session:
            db_hotspots = (
                session.execute(
                    select(SatelliteHotspotDB)
                    .where(
                        SatelliteHotspotDB.detected_at >= start_time,
                        SatelliteHotspotDB.detected_at <= end_time,
                    )
                    .order_by(SatelliteHotspotDB.detected_at.desc(), SatelliteHotspotDB.id.desc())
                )
                .scalars()
                .all()
            )
            return [self._to_domain_hotspot(db_hotspot) for db_hotspot in db_hotspots]

    def get_recent_hotspots(
        self,
        as_of: datetime,
        lookback_minutes: int,
    ) -> list[StoredSatelliteHotspot]:
        """Return persisted hotspots in the lookback window, newest first."""
        self._validate_recent_query(as_of, lookback_minutes)
        start_time = as_of - timedelta(minutes=lookback_minutes)

        with self._session_scope() as session:
            db_hotspots = (
                session.execute(
                    select(SatelliteHotspotDB)
                    .where(
                        SatelliteHotspotDB.detected_at >= start_time,
                        SatelliteHotspotDB.detected_at <= as_of,
                    )
                    .order_by(SatelliteHotspotDB.detected_at.desc(), SatelliteHotspotDB.id.desc())
                )
                .scalars()
                .all()
            )
            return [
                StoredSatelliteHotspot(id=db_hotspot.id, hotspot=self._to_domain_hotspot(db_hotspot))
                for db_hotspot in db_hotspots
            ]

    def get_by_id(self, hotspot_id: int) -> StoredSatelliteHotspot | None:
        """Return a persisted hotspot by database id, or None if absent."""
        self._validate_hotspot_id(hotspot_id)
        with self._session_scope() as session:
            db_hotspot = session.get(SatelliteHotspotDB, hotspot_id)
            if db_hotspot is None:
                return None
            return StoredSatelliteHotspot(id=db_hotspot.id, hotspot=self._to_domain_hotspot(db_hotspot))

    @staticmethod
    def _find_by_detection_key(session: Session, detection_key: str) -> SatelliteHotspotDB | None:
        return session.execute(
            select(SatelliteHotspotDB).where(SatelliteHotspotDB.detection_key == detection_key)
        ).scalar_one_or_none()

    @staticmethod
    def _to_domain_hotspot(db_hotspot: SatelliteHotspotDB) -> SatelliteHotspot:
        return SatelliteHotspot(
            latitude=db_hotspot.latitude,
            longitude=db_hotspot.longitude,
            detected_at=db_hotspot.detected_at,
            confidence=db_hotspot.confidence,
            frp=db_hotspot.frp,
            brightness=db_hotspot.brightness,
            satellite=db_hotspot.satellite,
            instrument=db_hotspot.instrument,
            day_night=db_hotspot.day_night,
        )

    @staticmethod
    def _generate_detection_key(hotspot: SatelliteHotspot) -> str:
        satellite = hotspot.satellite.strip() if hotspot.satellite is not None else _NONE_SATELLITE_SENTINEL
        canonical = "|".join(
            [
                repr(float(hotspot.latitude)),
                repr(float(hotspot.longitude)),
                hotspot.detected_at.isoformat(),
                satellite or _NONE_SATELLITE_SENTINEL,
            ]
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise SatelliteHotspotRepositoryError(f"Invalid limit: {limit!r}. Must be a positive integer.")

    @staticmethod
    def _validate_recent_query(as_of: datetime, lookback_minutes: int) -> None:
        if not isinstance(as_of, datetime):
            raise SatelliteHotspotRepositoryError(f"as_of must be a datetime, got {as_of!r}.")
        if (
            isinstance(lookback_minutes, bool)
            or not isinstance(lookback_minutes, int)
            or lookback_minutes <= 0
        ):
            raise SatelliteHotspotRepositoryError(
                f"lookback_minutes must be a positive integer, got {lookback_minutes!r}."
            )

    @staticmethod
    def _validate_hotspot_id(hotspot_id: int) -> None:
        if isinstance(hotspot_id, bool) or not isinstance(hotspot_id, int) or hotspot_id <= 0:
            raise SatelliteHotspotRepositoryError(f"hotspot_id must be a positive integer, got {hotspot_id!r}.")
