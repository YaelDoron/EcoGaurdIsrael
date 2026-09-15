"""Persistence layer for firefighting resource (fire truck) reference data.

FirefightingResourceRepository exposes read access to FirefightingResourceDB
rows (seeded from fabricated data - see backend/scripts/seed_fire_stations.py).
It is responsible only for persistence access: no geographic filtering,
station selection, or dispatch logic lives here.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.resource_status import ResourceStatus


class FirefightingResourceRepository:
    """Retrieves FirefightingResourceDB rows via SQLAlchemy.

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

    def get_available_resources(self, station_ids: list[str]) -> list[FirefightingResourceDB]:
        """Return AVAILABLE resources attached to any of the given station ids.

        Returns an empty list immediately, without querying the database, if
        `station_ids` is empty.
        """
        if not station_ids:
            return []

        with self._session_scope() as session:
            return list(
                session.execute(
                    select(FirefightingResourceDB)
                    .where(
                        FirefightingResourceDB.station_id.in_(station_ids),
                        FirefightingResourceDB.status == ResourceStatus.AVAILABLE,
                    )
                    .order_by(FirefightingResourceDB.station_id, FirefightingResourceDB.id)
                )
                .scalars()
                .all()
            )

    def get_resources_for_stations(self, station_ids: list[str]) -> list[FirefightingResourceDB]:
        """Return every resource (any status) attached to any of the given station ids.

        Returns an empty list immediately, without querying the database, if
        `station_ids` is empty.
        """
        if not station_ids:
            return []

        with self._session_scope() as session:
            return list(
                session.execute(
                    select(FirefightingResourceDB)
                    .where(FirefightingResourceDB.station_id.in_(station_ids))
                    .order_by(FirefightingResourceDB.station_id, FirefightingResourceDB.id)
                )
                .scalars()
                .all()
            )

    def set_statuses(self, resource_ids: list[str], status: ResourceStatus) -> int:
        """Set the status of the given resources; returns the number of rows updated.

        Returns 0 immediately, without querying the database, if
        `resource_ids` is empty.
        """
        if not resource_ids:
            return 0

        with self._session_scope() as session:
            result = session.execute(
                update(FirefightingResourceDB)
                .where(FirefightingResourceDB.id.in_(resource_ids))
                .values(status=status)
            )
            return result.rowcount or 0
