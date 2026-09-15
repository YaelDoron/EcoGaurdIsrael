"""Persistence layer for generated response-target snapshots."""
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
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models.response_target import ResponseTarget
from src.models.response_target_set import ResponseTargetSet
from src.models.response_target_type import ResponseTargetType
from src.repositories.exceptions import ResponseTargetRepositoryError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredResponseTarget:
    """Persisted response target plus database identity and target order."""

    id: int
    target_order: int
    target: ResponseTarget


@dataclass(frozen=True)
class StoredResponseTargetSet:
    """Persisted response target set plus database identity and ordered targets."""

    id: int
    target_set: ResponseTargetSet
    targets: tuple[StoredResponseTarget, ...] = ()


class ResponseTargetRepository:
    """Persists response-target snapshots via SQLAlchemy."""

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

    def save_target_set(self, target_set: ResponseTargetSet) -> StoredResponseTargetSet:
        """Atomically persist a complete ordered response-target snapshot."""
        if not isinstance(target_set, ResponseTargetSet):
            raise ResponseTargetRepositoryError(f"target_set must be a ResponseTargetSet, got {target_set!r}")

        with self._session_scope() as session:
            self._validate_predicted_sources(session, target_set)
            db_set = ResponseTargetSetDB(
                fire_event_id=target_set.fire_event_id,
                generated_at=target_set.generated_at,
                methodology=target_set.methodology,
                methodology_version=target_set.methodology_version,
            )
            session.add(db_set)
            try:
                session.flush()
                for target_order, target in enumerate(target_set.targets):
                    session.add(self._to_db_target(db_set.id, target_order, target))
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise ResponseTargetRepositoryError("Response-target set persistence failed.") from exc

            logger.info(
                "Stored response-target set %s for FireEvent %s with %s targets",
                db_set.id,
                target_set.fire_event_id,
                len(target_set.targets),
            )
            return self._to_stored_target_set(db_set)

    def get_by_id(self, target_set_id: int) -> StoredResponseTargetSet | None:
        """Return a persisted response-target set by database id, or None if absent."""
        self._validate_target_set_id(target_set_id)
        with self._session_scope() as session:
            db_set = self._get_db_target_set(session, target_set_id)
            return self._to_stored_target_set(db_set) if db_set is not None else None

    def get_latest_for_event_as_of(
        self,
        fire_event_id: int,
        as_of: datetime,
    ) -> StoredResponseTargetSet | None:
        """Return the latest generated target set for a FireEvent at or before `as_of`."""
        self._validate_fire_event_id(fire_event_id)
        self._validate_aware_datetime("as_of", as_of)
        with self._session_scope() as session:
            db_set = (
                session.execute(
                    select(ResponseTargetSetDB)
                    .options(selectinload(ResponseTargetSetDB.targets))
                    .where(
                        ResponseTargetSetDB.fire_event_id == fire_event_id,
                        ResponseTargetSetDB.generated_at <= as_of,
                    )
                    .order_by(ResponseTargetSetDB.generated_at.desc(), ResponseTargetSetDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_target_set(db_set) if db_set is not None else None

    def get_history_for_event(self, fire_event_id: int) -> tuple[StoredResponseTargetSet, ...]:
        """Return all persisted response-target sets for a FireEvent, newest first."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            db_sets = (
                session.execute(
                    select(ResponseTargetSetDB)
                    .options(selectinload(ResponseTargetSetDB.targets))
                    .where(ResponseTargetSetDB.fire_event_id == fire_event_id)
                    .order_by(ResponseTargetSetDB.generated_at.desc(), ResponseTargetSetDB.id.desc())
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_target_set(db_set) for db_set in db_sets)

    def _validate_predicted_sources(self, session: Session, target_set: ResponseTargetSet) -> None:
        for target in target_set.targets:
            if target.target_type is not ResponseTargetType.PREDICTED_RISK:
                continue

            db_prediction = session.get(FireSpreadPredictionDB, target.spread_prediction_id)
            if db_prediction is None:
                raise ResponseTargetRepositoryError(
                    f"Spread prediction {target.spread_prediction_id!r} was not found."
                )
            if db_prediction.fire_event_id != target.fire_event_id:
                raise ResponseTargetRepositoryError(
                    "spread prediction fire_event_id must match target fire_event_id, got "
                    f"{db_prediction.fire_event_id!r} for target {target.fire_event_id!r}."
                )

            db_cell = session.get(FireSpreadPredictionCellDB, target.spread_prediction_cell_id)
            if db_cell is None:
                raise ResponseTargetRepositoryError(
                    f"Spread prediction cell {target.spread_prediction_cell_id!r} was not found."
                )
            if db_cell.prediction_id != target.spread_prediction_id:
                raise ResponseTargetRepositoryError(
                    "spread prediction cell must belong to the referenced spread prediction, got "
                    f"cell prediction_id={db_cell.prediction_id!r} for prediction {target.spread_prediction_id!r}."
                )

    @staticmethod
    def _to_db_target(response_target_set_id: int, target_order: int, target: ResponseTarget) -> ResponseTargetDB:
        return ResponseTargetDB(
            response_target_set_id=response_target_set_id,
            fire_event_id=target.fire_event_id,
            target_order=target_order,
            target_type=target.target_type.value,
            latitude=target.latitude,
            longitude=target.longitude,
            priority_score=target.priority_score,
            prediction_horizon_minutes=target.prediction_horizon_minutes,
            spread_prediction_id=target.spread_prediction_id,
            spread_prediction_cell_id=target.spread_prediction_cell_id,
        )

    @staticmethod
    def _get_db_target_set(session: Session, target_set_id: int) -> ResponseTargetSetDB | None:
        return (
            session.execute(
                select(ResponseTargetSetDB)
                .options(selectinload(ResponseTargetSetDB.targets))
                .where(ResponseTargetSetDB.id == target_set_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _to_stored_target_set(cls, db_set: ResponseTargetSetDB) -> StoredResponseTargetSet:
        stored_targets = tuple(
            StoredResponseTarget(
                id=db_target.id,
                target_order=db_target.target_order,
                target=ResponseTarget(
                    fire_event_id=db_target.fire_event_id,
                    target_type=ResponseTargetType(db_target.target_type),
                    latitude=db_target.latitude,
                    longitude=db_target.longitude,
                    priority_score=db_target.priority_score,
                    prediction_horizon_minutes=db_target.prediction_horizon_minutes,
                    spread_prediction_id=db_target.spread_prediction_id,
                    spread_prediction_cell_id=db_target.spread_prediction_cell_id,
                ),
            )
            for db_target in sorted(db_set.targets, key=lambda target: (target.target_order, target.id))
        )
        target_set = ResponseTargetSet(
            fire_event_id=db_set.fire_event_id,
            generated_at=cls._ensure_aware_datetime(db_set.generated_at),
            methodology=db_set.methodology,
            methodology_version=db_set.methodology_version,
            targets=tuple(stored_target.target for stored_target in stored_targets),
        )
        return StoredResponseTargetSet(
            id=db_set.id,
            target_set=target_set,
            targets=stored_targets,
        )

    @staticmethod
    def _validate_target_set_id(target_set_id: int) -> None:
        if isinstance(target_set_id, bool) or not isinstance(target_set_id, int) or target_set_id <= 0:
            raise ResponseTargetRepositoryError(f"target_set_id must be a positive integer, got {target_set_id!r}.")

    @staticmethod
    def _validate_fire_event_id(fire_event_id: int) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ResponseTargetRepositoryError(f"fire_event_id must be a positive integer, got {fire_event_id!r}.")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ResponseTargetRepositoryError(f"{field_name} must be a timezone-aware datetime, got {value!r}.")

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
