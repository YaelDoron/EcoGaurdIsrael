"""Canonical FireEvent lifecycle transitions (Stage 1.1 of the Global
Multi-Incident Optimizer refactor).

Before this service existed, `FireEventRepository.update_event()` was the
only production path that ever wrote `FireEvent.status`, and it had exactly
one production caller (`FireDetectionAgent._update_existing_event`, moving
only SUSPECTED<->CONFIRMED as a side effect of re-evaluating combined
evidence). No production path anywhere in this codebase ever transitioned a
FireEvent to RESOLVED or DISMISSED (see this refactor's Stage 1.1 report,
Task 1, for the full audit) - Stage 1's own defensive cleanup in
ResponsePlanningRefreshOrchestrator was written to be safe if that ever
happened, but nothing ever made it happen.

FireEventLifecycleService is the canonical entry point for the two inactive
transitions this refactor cares about: resolve_event and dismiss_event.
Each atomically (one session, one commit-or-rollback):
  (a) locks the FireEvent row;
  (b) writes the new status + updated_at;
  (c) releases every ResourceCommitment currently owned by that FireEvent
      (Task 10's invariant: an inactive FireEvent must own zero
      commitments, immediately, not only once some future planning
      refresh happens to notice).
Historical ResponsePlan/ResponseAction/ResponsePlanPlanningState/
RoutePlanningRun/ResponseTargetSet rows are never touched - only current
resource OWNERSHIP (ResourceCommitment) is a live-state concept here.

Deliberately narrow: only resolve_event/dismiss_event exist. This service
does not validate or restrict which transitions are "allowed" beyond a
FireEvent existing - update_event() already enforced no transition rules
at all, and Task 3 of this stage is explicit that new lifecycle rules must
not be invented here. Calling resolve_event/dismiss_event on an
already-inactive FireEvent is safe and idempotent: the status write is a
no-op value-wise and the commitment release naturally deletes zero rows.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_event_db import FireEventDB
from src.models.fire_event_status import FireEventStatus
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository


class FireEventLifecycleService:
    """Atomically transitions a FireEvent to an inactive status and releases its commitments."""

    def __init__(
        self,
        fire_event_repository: FireEventRepository | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()
        self._session_factory = session_factory or get_session_factory()

    def resolve_event(self, fire_event_id: int, *, as_of: datetime) -> StoredFireEvent:
        """Transition a FireEvent to RESOLVED and release its resource commitments atomically."""
        return self._transition_to_inactive(fire_event_id, FireEventStatus.RESOLVED, as_of)

    def dismiss_event(self, fire_event_id: int, *, as_of: datetime) -> StoredFireEvent:
        """Transition a FireEvent to DISMISSED and release its resource commitments atomically."""
        return self._transition_to_inactive(fire_event_id, FireEventStatus.DISMISSED, as_of)

    def _transition_to_inactive(
        self, fire_event_id: int, new_status: FireEventStatus, as_of: datetime
    ) -> StoredFireEvent:
        self._validate_fire_event_id(fire_event_id)
        self._validate_as_of(as_of)

        session = self._session_factory()
        try:
            db_event = session.execute(
                select(FireEventDB).where(FireEventDB.id == fire_event_id).with_for_update()
            ).scalar_one_or_none()
            if db_event is None:
                raise ValueError(f"FireEvent {fire_event_id!r} was not found.")

            db_event.status = new_status.value
            db_event.updated_at = as_of
            session.flush()

            self._resource_commitment_repository.release_for_fire_event_in_session(session, fire_event_id)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return self._fire_event_repository.get_by_id(fire_event_id)

    @staticmethod
    def _validate_fire_event_id(fire_event_id: object) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}.")

    @staticmethod
    def _validate_as_of(as_of: object) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}.")
