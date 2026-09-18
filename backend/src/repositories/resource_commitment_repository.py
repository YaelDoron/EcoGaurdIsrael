"""Persistence layer for persistent resource ownership (Stage 1 of the
Global Multi-Incident Optimizer refactor).

Simple standalone reads (get_by_resource_id, get_for_fire_event,
get_resource_ids_for_other_active_events, release_for_fire_event) each own
their own session/transaction, matching this project's usual repository
convention. `replace_commitments_for_plan` is different by design: it
accepts a CALLER-OWNED session (mirrors RoadNetworkRepository.save_network's
own precedent - see operational_context_service.py) and does not commit -
it exists to be composed into a larger atomic transaction owned by
ResponsePlanActivationService (row locking + commitment replacement + the
ResponsePlanPlanningState sidecar insert all committing together, or all
rolling back together).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_event_db import FireEventDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.models.dispatch_state import DispatchState
from src.models.fire_event_status import FireEventStatus
from src.models.resource_commitment import ResourceCommitment
from src.repositories.exceptions import ResourceCommitmentRepositoryError

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}


class ResourceCommitmentRepository:
    """Reads/writes ResourceCommitment rows (see this module's docstring for the session-ownership split)."""

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

    def get_by_resource_id(self, resource_id: str) -> ResourceCommitment | None:
        """Return the current commitment for one resource, or None if it is uncommitted."""
        self._validate_resource_id(resource_id)
        with self._session_scope() as session:
            db_commitment = session.get(ResourceCommitmentDB, resource_id)
            return self._to_domain(db_commitment) if db_commitment is not None else None

    def get_for_fire_event(self, fire_event_id: int) -> tuple[ResourceCommitment, ...]:
        """Return every resource currently committed to this FireEvent, ordered by resource_id."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            db_commitments = (
                session.execute(
                    select(ResourceCommitmentDB)
                    .where(ResourceCommitmentDB.fire_event_id == fire_event_id)
                    .order_by(ResourceCommitmentDB.resource_id)
                )
                .scalars()
                .all()
            )
            return tuple(self._to_domain(row) for row in db_commitments)

    def get_for_resource_ids(self, resource_ids: Iterable[str]) -> tuple[ResourceCommitment, ...]:
        """Return the current commitment for every one of the given resource_ids that has one.

        Batched (one IN query) for GlobalCandidateCollector's commitment-
        visibility step (Stage 3, Task 9) - resources with no commitment
        simply contribute nothing here, they are not an error.
        """
        ids = self._normalize_resource_id_list(resource_ids)
        if not ids:
            return ()
        with self._session_scope() as session:
            db_commitments = (
                session.execute(
                    select(ResourceCommitmentDB)
                    .where(ResourceCommitmentDB.resource_id.in_(ids))
                    .order_by(ResourceCommitmentDB.resource_id)
                )
                .scalars()
                .all()
            )
            return tuple(self._to_domain(row) for row in db_commitments)

    def get_for_fire_events(self, fire_event_ids: Iterable[int]) -> tuple[ResourceCommitment, ...]:
        """Return every commitment owned by any of the given FireEvents, ordered by resource_id.

        Batched (one IN query) for GlobalPlanningSnapshot capture (Stage 2,
        Task 6) rather than one get_for_fire_event call per event.
        """
        ids = self._normalize_fire_event_id_list(fire_event_ids)
        if not ids:
            return ()
        with self._session_scope() as session:
            db_commitments = (
                session.execute(
                    select(ResourceCommitmentDB)
                    .where(ResourceCommitmentDB.fire_event_id.in_(ids))
                    .order_by(ResourceCommitmentDB.resource_id)
                )
                .scalars()
                .all()
            )
            return tuple(self._to_domain(row) for row in db_commitments)

    def get_resource_ids_for_other_active_events(self, excluded_fire_event_id: int) -> frozenset[str]:
        """Return resource ids committed to any OTHER currently-active FireEvent.

        Defensive on FireEvent activity: joins against fire_events.status
        rather than assuming every commitment row necessarily belongs to a
        still-active event (release_for_fire_event/Task 14's lifecycle hook
        should already guarantee this in the steady state, but this read
        does not depend on that having run promptly).
        """
        self._validate_fire_event_id(excluded_fire_event_id)
        with self._session_scope() as session:
            resource_ids = (
                session.execute(
                    select(ResourceCommitmentDB.resource_id)
                    .join(FireEventDB, FireEventDB.id == ResourceCommitmentDB.fire_event_id)
                    .where(
                        ResourceCommitmentDB.fire_event_id != excluded_fire_event_id,
                        FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES),
                    )
                )
                .scalars()
                .all()
            )
            return frozenset(resource_ids)

    def release_for_fire_event(self, fire_event_id: int) -> int:
        """Remove every commitment currently owned by this FireEvent. Returns the count removed."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            return self._release_in_session(session, fire_event_id)

    def release_for_fire_event_in_session(self, session: Session, fire_event_id: int) -> int:
        """Same release as release_for_fire_event, composed into a CALLER-OWNED
        session/transaction (Stage 1.1: FireEventLifecycleService uses this to
        make the status transition and the commitment release atomic - both
        commit or both roll back together). Does NOT commit or close the
        session - see this module's docstring."""
        self._validate_fire_event_id(fire_event_id)
        return self._release_in_session(session, fire_event_id)

    @staticmethod
    def _release_in_session(session: Session, fire_event_id: int) -> int:
        result = session.execute(
            delete(ResourceCommitmentDB).where(ResourceCommitmentDB.fire_event_id == fire_event_id)
        )
        return result.rowcount or 0

    def replace_commitments_for_plan(
        self,
        session: Session,
        *,
        fire_event_id: int,
        response_plan_id: int,
        resource_ids: Iterable[str],
        committed_at: datetime,
        dispatch_state: DispatchState = DispatchState.PLANNED,
    ) -> None:
        """Release this FireEvent's existing commitments and create new ones for `resource_ids`.

        Caller-owned session: does NOT commit or close it - see this
        module's docstring. Callers are responsible for already having
        locked the relevant resource rows and verified none of
        `resource_ids` is committed to a DIFFERENT FireEvent before calling
        this (ResponsePlanActivationService does both).

        `dispatch_state` (Stage 6) defaults to PLANNED, preserving the
        legacy per-event ResponsePlanActivationService's existing behavior
        unchanged. GlobalResponsePlanActivationService passes DISPATCHED
        explicitly for resources the Global GA result places into an
        activating plan - see src/models/dispatch_state.py.
        """
        self._validate_fire_event_id(fire_event_id)
        self._validate_response_plan_id(response_plan_id)
        ids = self._normalize_resource_ids(resource_ids)
        self._validate_aware_datetime("committed_at", committed_at)
        if not isinstance(dispatch_state, DispatchState):
            raise ResourceCommitmentRepositoryError(f"dispatch_state must be a DispatchState, got {dispatch_state!r}.")

        session.execute(delete(ResourceCommitmentDB).where(ResourceCommitmentDB.fire_event_id == fire_event_id))
        for resource_id in ids:
            session.add(
                ResourceCommitmentDB(
                    resource_id=resource_id,
                    fire_event_id=fire_event_id,
                    response_plan_id=response_plan_id,
                    committed_at=committed_at,
                    dispatch_state=dispatch_state.value,
                )
            )
        session.flush()

    @staticmethod
    def _to_domain(db_commitment: ResourceCommitmentDB) -> ResourceCommitment:
        committed_at = db_commitment.committed_at
        if committed_at.tzinfo is None:
            committed_at = committed_at.replace(tzinfo=timezone.utc)
        return ResourceCommitment(
            resource_id=db_commitment.resource_id,
            fire_event_id=db_commitment.fire_event_id,
            response_plan_id=db_commitment.response_plan_id,
            committed_at=committed_at,
            dispatch_state=DispatchState(db_commitment.dispatch_state),
        )

    @staticmethod
    def _validate_resource_id(resource_id: object) -> None:
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise ResourceCommitmentRepositoryError(f"resource_id must be a non-empty string, got {resource_id!r}.")

    @staticmethod
    def _normalize_resource_id_list(resource_ids: Iterable[str]) -> tuple[str, ...]:
        try:
            values = tuple(resource_ids)
        except TypeError as exc:
            raise ResourceCommitmentRepositoryError("resource_ids must be iterable.") from exc
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ResourceCommitmentRepositoryError(
                    f"resource_ids must contain non-empty strings, got {value!r}."
                )
        return tuple(sorted(set(values)))

    @staticmethod
    def _normalize_fire_event_id_list(fire_event_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            values = tuple(fire_event_ids)
        except TypeError as exc:
            raise ResourceCommitmentRepositoryError("fire_event_ids must be iterable.") from exc
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResourceCommitmentRepositoryError(
                    f"fire_event_ids must contain positive integer ids, got {value!r}."
                )
        return tuple(sorted(set(values)))

    @staticmethod
    def _validate_fire_event_id(fire_event_id: object) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ResourceCommitmentRepositoryError(
                f"fire_event_id must be a positive integer, got {fire_event_id!r}."
            )

    @staticmethod
    def _validate_response_plan_id(response_plan_id: object) -> None:
        if isinstance(response_plan_id, bool) or not isinstance(response_plan_id, int) or response_plan_id <= 0:
            raise ResourceCommitmentRepositoryError(
                f"response_plan_id must be a positive integer, got {response_plan_id!r}."
            )

    @staticmethod
    def _normalize_resource_ids(resource_ids: Iterable[str]) -> tuple[str, ...]:
        try:
            values = tuple(resource_ids)
        except TypeError as exc:
            raise ResourceCommitmentRepositoryError("resource_ids must be iterable.") from exc
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ResourceCommitmentRepositoryError(
                    f"resource_ids must contain non-empty strings, got {value!r}."
                )
        return tuple(sorted(set(values)))

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ResourceCommitmentRepositoryError(
                f"{field_name} must be a timezone-aware datetime, got {value!r}."
            )
