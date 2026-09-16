"""Persistence layer for routing runs (Epic 5 / User Story 5.1)."""
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
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.models.routing import RoutePlanningRun, RouteResult, RouteStatus, StoredRouteResult
from src.repositories.exceptions import RoutePlanningRepositoryError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredRoutePlanningRun:
    """Persisted routing run plus database identity and its stored route results."""

    id: int
    run: RoutePlanningRun
    routes: tuple[StoredRouteResult, ...] = ()


class RoutePlanningRepository:
    """Persists routing runs (and their per-resource/target RouteResults) via SQLAlchemy."""

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

    def save_run(self, run: RoutePlanningRun) -> StoredRoutePlanningRun:
        """Atomically persist a complete routing run and every RouteResult it contains."""
        if not isinstance(run, RoutePlanningRun):
            raise RoutePlanningRepositoryError(f"run must be a RoutePlanningRun, got {run!r}")

        with self._session_scope() as session:
            self._validate_response_target_set(session, run)
            self._validate_response_targets(session, run)

            db_run = RoutePlanningRunDB(
                fire_event_id=run.fire_event_id,
                response_target_set_id=run.response_target_set_id,
                planned_at=run.planned_at,
                methodology=run.methodology,
                methodology_version=run.methodology_version,
                resource_ids=list(run.resource_ids),
            )
            session.add(db_run)
            try:
                session.flush()
                for route in run.routes:
                    session.add(self._to_db_route(db_run.id, route))
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise RoutePlanningRepositoryError("Routing run persistence failed.") from exc

            logger.info(
                "Stored routing run %s for FireEvent %s with %s route result(s)",
                db_run.id,
                run.fire_event_id,
                len(run.routes),
            )
            return self._to_stored_run(db_run)

    def get_by_id(self, run_id: int) -> StoredRoutePlanningRun | None:
        """Return a persisted routing run by database id, or None if absent."""
        self._validate_positive_int("run_id", run_id)
        with self._session_scope() as session:
            db_run = self._get_db_run(session, run_id)
            return self._to_stored_run(db_run) if db_run is not None else None

    def get_latest_for_event_as_of(
        self,
        fire_event_id: int,
        as_of: datetime,
    ) -> StoredRoutePlanningRun | None:
        """Return the latest routing run for a FireEvent planned at or before `as_of`."""
        self._validate_positive_int("fire_event_id", fire_event_id)
        self._validate_aware_datetime("as_of", as_of)
        with self._session_scope() as session:
            db_run = (
                session.execute(
                    select(RoutePlanningRunDB)
                    .options(selectinload(RoutePlanningRunDB.routes))
                    .where(
                        RoutePlanningRunDB.fire_event_id == fire_event_id,
                        RoutePlanningRunDB.planned_at <= as_of,
                    )
                    .order_by(RoutePlanningRunDB.planned_at.desc(), RoutePlanningRunDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_run(db_run) if db_run is not None else None

    def get_history_for_event(self, fire_event_id: int) -> tuple[StoredRoutePlanningRun, ...]:
        """Return all persisted routing runs for a FireEvent, newest first."""
        self._validate_positive_int("fire_event_id", fire_event_id)
        with self._session_scope() as session:
            db_runs = (
                session.execute(
                    select(RoutePlanningRunDB)
                    .options(selectinload(RoutePlanningRunDB.routes))
                    .where(RoutePlanningRunDB.fire_event_id == fire_event_id)
                    .order_by(RoutePlanningRunDB.planned_at.desc(), RoutePlanningRunDB.id.desc())
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_run(db_run) for db_run in db_runs)

    def _validate_response_target_set(self, session: Session, run: RoutePlanningRun) -> None:
        db_target_set = session.get(ResponseTargetSetDB, run.response_target_set_id)
        if db_target_set is None:
            raise RoutePlanningRepositoryError(f"ResponseTargetSet {run.response_target_set_id!r} was not found.")
        if db_target_set.fire_event_id != run.fire_event_id:
            raise RoutePlanningRepositoryError(
                "run response_target_set_id must belong to the run's fire_event, got "
                f"target set fire_event_id={db_target_set.fire_event_id!r} for run "
                f"fire_event_id={run.fire_event_id!r}."
            )

    def _validate_response_targets(self, session: Session, run: RoutePlanningRun) -> None:
        for route in run.routes:
            db_target = session.get(ResponseTargetDB, route.response_target_id)
            if db_target is None:
                raise RoutePlanningRepositoryError(f"ResponseTarget {route.response_target_id!r} was not found.")
            if db_target.response_target_set_id != run.response_target_set_id:
                raise RoutePlanningRepositoryError(
                    "route response_target_id must belong to the run's response_target_set, got "
                    f"target response_target_set_id={db_target.response_target_set_id!r} for run "
                    f"response_target_set_id={run.response_target_set_id!r}."
                )

    @staticmethod
    def _to_db_route(route_planning_run_id: int, route: RouteResult) -> RouteResultDB:
        return RouteResultDB(
            route_planning_run_id=route_planning_run_id,
            resource_id=route.resource_id,
            response_target_id=route.response_target_id,
            status=route.status.value,
            source_node_id=route.source_node_id,
            target_node_id=route.target_node_id,
            node_path=list(route.node_path),
            distance_meters=route.distance_meters,
            travel_time_seconds=route.travel_time_seconds,
        )

    @staticmethod
    def _get_db_run(session: Session, run_id: int) -> RoutePlanningRunDB | None:
        return (
            session.execute(
                select(RoutePlanningRunDB)
                .options(selectinload(RoutePlanningRunDB.routes))
                .where(RoutePlanningRunDB.id == run_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _to_stored_run(cls, db_run: RoutePlanningRunDB) -> StoredRoutePlanningRun:
        stored_routes = tuple(
            StoredRouteResult(id=db_route.id, route_result=cls._to_route_result(db_route))
            for db_route in sorted(db_run.routes, key=lambda db_route: db_route.id)
        )
        run = RoutePlanningRun(
            fire_event_id=db_run.fire_event_id,
            response_target_set_id=db_run.response_target_set_id,
            planned_at=cls._ensure_aware_datetime(db_run.planned_at),
            methodology=db_run.methodology,
            methodology_version=db_run.methodology_version,
            resource_ids=tuple(db_run.resource_ids),
            routes=tuple(stored_route.route_result for stored_route in stored_routes),
        )
        return StoredRoutePlanningRun(id=db_run.id, run=run, routes=stored_routes)

    @staticmethod
    def _to_route_result(db_route: RouteResultDB) -> RouteResult:
        return RouteResult(
            resource_id=db_route.resource_id,
            response_target_id=db_route.response_target_id,
            status=RouteStatus(db_route.status),
            source_node_id=db_route.source_node_id,
            target_node_id=db_route.target_node_id,
            node_path=tuple(db_route.node_path),
            distance_meters=db_route.distance_meters,
            travel_time_seconds=db_route.travel_time_seconds,
        )

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RoutePlanningRepositoryError(f"{field_name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise RoutePlanningRepositoryError(f"{field_name} must be a timezone-aware datetime, got {value!r}.")

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
