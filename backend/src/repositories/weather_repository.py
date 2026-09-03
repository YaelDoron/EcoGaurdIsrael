"""Persistence layer for weather data.

WeatherRepository bridges the Task 2 domain dataclasses (WeatherStation,
WeatherObservation) and the SQLAlchemy ORM models (WeatherStationDB,
WeatherObservationDB). It is responsible ONLY for persistence: it does not
call IMS, does not know raw IMS channel names, does not calculate wildfire
risk, and does not schedule or orchestrate anything.

Every public method takes and returns Task 2 domain models, never the
SQLAlchemy ORM models - callers of this repository never need to know
SQLAlchemy exists.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import WeatherRepositoryError, WeatherStationNotStoredError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SaveObservationResult:
    """Result of WeatherRepository.save_observation().

    `is_duplicate` is True when an observation already existed for the same
    (station, timestamp); in that case `observation` is the *existing*
    stored row, not the newly-submitted one. This gives callers (e.g.
    WeatherAgent) an explicit signal for duplicate handling, rather than
    having to compare field values.
    """

    observation: WeatherObservation
    is_duplicate: bool


class WeatherRepository:
    """Persists and retrieves WeatherStation/WeatherObservation via SQLAlchemy.

    Accepts an optional `session_factory` for dependency injection (e.g. a
    SQLite in-memory sessionmaker in tests); defaults to the process-wide
    Neon/PostgreSQL session factory.
    """

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

    # ------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------

    def save_station(self, station: WeatherStation) -> WeatherStation:
        """Insert or update a weather station (upsert), keyed by external_station_id."""
        with self._session_scope() as session:
            db_station = self._find_station_by_external_id(session, station.external_station_id)

            if db_station is None:
                db_station = WeatherStationDB(
                    external_station_id=station.external_station_id,
                    name=station.name,
                    latitude=station.latitude,
                    longitude=station.longitude,
                    region_id=station.region_id,
                    active=station.active,
                )
                session.add(db_station)
                logger.info("Inserted weather station %s", station.external_station_id)
            else:
                db_station.name = station.name
                db_station.latitude = station.latitude
                db_station.longitude = station.longitude
                db_station.region_id = station.region_id
                db_station.active = station.active
                logger.info("Updated weather station %s", station.external_station_id)

            session.flush()
            return self._to_domain_station(db_station)

    def get_station_by_external_id(self, external_station_id: int) -> WeatherStation | None:
        """Return the stored station for the given IMS station id, or None if not found."""
        self._validate_external_station_id(external_station_id)
        with self._session_scope() as session:
            db_station = self._find_station_by_external_id(session, external_station_id)
            return self._to_domain_station(db_station) if db_station is not None else None

    def get_all_stations(self) -> list[WeatherStation]:
        """Return all stored stations, ordered deterministically by external_station_id."""
        with self._session_scope() as session:
            db_stations = (
                session.execute(select(WeatherStationDB).order_by(WeatherStationDB.external_station_id))
                .scalars()
                .all()
            )
            return [self._to_domain_station(db_station) for db_station in db_stations]

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def save_observation(self, observation: WeatherObservation) -> SaveObservationResult:
        """Insert a weather observation, keyed uniquely by (station, timestamp).

        Raises WeatherStationNotStoredError if `observation.station_external_id`
        has no corresponding stored station - this repository never invents
        station metadata.

        If an observation already exists for the same station and timestamp,
        no duplicate row is created; the existing stored observation is
        returned with `is_duplicate=True`. This covers both the ordinary
        check-then-insert path and a concurrent-write race (caught via
        IntegrityError from the database's own unique constraint).
        """
        with self._session_scope() as session:
            db_station = self._find_station_by_external_id(session, observation.station_external_id)
            if db_station is None:
                raise WeatherStationNotStoredError(
                    f"No stored station with external_station_id={observation.station_external_id}."
                )

            existing = self._find_observation(session, db_station.id, observation.timestamp)
            if existing is not None:
                logger.info(
                    "Skipped duplicate weather observation for station %s at %s",
                    observation.station_external_id,
                    observation.timestamp,
                )
                return SaveObservationResult(
                    observation=self._to_domain_observation(existing, db_station.external_station_id),
                    is_duplicate=True,
                )

            db_observation = WeatherObservationDB(
                station_id=db_station.id,
                timestamp=observation.timestamp,
                temperature=observation.temperature,
                relative_humidity=observation.relative_humidity,
                wind_speed=observation.wind_speed,
                wind_direction=observation.wind_direction,
                wind_gust=observation.wind_gust,
                rainfall=observation.rainfall,
            )
            session.add(db_observation)

            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                logger.info(
                    "Duplicate weather observation detected on write for station %s at %s",
                    observation.station_external_id,
                    observation.timestamp,
                )
                existing = self._find_observation(session, db_station.id, observation.timestamp)
                if existing is None:
                    raise
                return SaveObservationResult(
                    observation=self._to_domain_observation(existing, db_station.external_station_id),
                    is_duplicate=True,
                )

            logger.info(
                "Stored weather observation for station %s at %s",
                observation.station_external_id,
                observation.timestamp,
            )
            return SaveObservationResult(
                observation=self._to_domain_observation(db_observation, db_station.external_station_id),
                is_duplicate=False,
            )

    def get_latest_observation(self, external_station_id: int) -> WeatherObservation | None:
        """Return the most recent observation for a station, or None if there are none."""
        self._validate_external_station_id(external_station_id)
        with self._session_scope() as session:
            db_station = self._find_station_by_external_id(session, external_station_id)
            if db_station is None:
                return None

            db_observation = (
                session.execute(
                    select(WeatherObservationDB)
                    .where(WeatherObservationDB.station_id == db_station.id)
                    .order_by(WeatherObservationDB.timestamp.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if db_observation is None:
                return None

            return self._to_domain_observation(db_observation, db_station.external_station_id)

    def get_observations_for_station(
        self, external_station_id: int, limit: int | None = None
    ) -> list[WeatherObservation]:
        """Return observations for a station, newest first (optional)."""
        self._validate_external_station_id(external_station_id)
        with self._session_scope() as session:
            db_station = self._find_station_by_external_id(session, external_station_id)
            if db_station is None:
                return []

            query = (
                select(WeatherObservationDB)
                .where(WeatherObservationDB.station_id == db_station.id)
                .order_by(WeatherObservationDB.timestamp.desc())
            )
            if limit is not None:
                query = query.limit(limit)

            db_observations = session.execute(query).scalars().all()
            return [
                self._to_domain_observation(db_observation, db_station.external_station_id)
                for db_observation in db_observations
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_station_by_external_id(session: Session, external_station_id: int) -> WeatherStationDB | None:
        return session.execute(
            select(WeatherStationDB).where(WeatherStationDB.external_station_id == external_station_id)
        ).scalar_one_or_none()

    @staticmethod
    def _find_observation(
        session: Session, station_id: int, timestamp: datetime
    ) -> WeatherObservationDB | None:
        return session.execute(
            select(WeatherObservationDB).where(
                WeatherObservationDB.station_id == station_id,
                WeatherObservationDB.timestamp == timestamp,
            )
        ).scalar_one_or_none()

    @staticmethod
    def _to_domain_station(db_station: WeatherStationDB) -> WeatherStation:
        return WeatherStation(
            external_station_id=db_station.external_station_id,
            name=db_station.name,
            latitude=db_station.latitude,
            longitude=db_station.longitude,
            region_id=db_station.region_id,
            active=db_station.active,
        )

    @staticmethod
    def _to_domain_observation(
        db_observation: WeatherObservationDB, external_station_id: int
    ) -> WeatherObservation:
        return WeatherObservation(
            station_external_id=external_station_id,
            timestamp=db_observation.timestamp,
            temperature=db_observation.temperature,
            relative_humidity=db_observation.relative_humidity,
            wind_speed=db_observation.wind_speed,
            wind_direction=db_observation.wind_direction,
            wind_gust=db_observation.wind_gust,
            rainfall=db_observation.rainfall,
        )

    @staticmethod
    def _validate_external_station_id(external_station_id: int) -> None:
        if (
            isinstance(external_station_id, bool)
            or not isinstance(external_station_id, int)
            or external_station_id <= 0
        ):
            raise WeatherRepositoryError(
                f"Invalid external_station_id: {external_station_id!r}. Must be a positive integer."
            )
