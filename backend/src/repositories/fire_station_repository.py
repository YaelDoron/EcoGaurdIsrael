"""Persistence layer for fire station reference data.

FireStationRepository exposes read access to FireStationDB rows (seeded from
governmental open data - see backend/scripts/seed_fire_stations.py). It is
responsible only for persistence access: no geographic filtering,
resource-allocation, or dispatch logic lives here.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_station_db import FireStationDB


class FireStationRepository:
    """Retrieves FireStationDB rows via SQLAlchemy.

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

    def get_all_stations(self) -> list[FireStationDB]:
        """Return every stored fire station, ordered deterministically by id."""
        with self._session_scope() as session:
            return list(
                session.execute(select(FireStationDB).order_by(FireStationDB.id)).scalars().all()
            )
