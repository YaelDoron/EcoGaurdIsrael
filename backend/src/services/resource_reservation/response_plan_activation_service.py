"""Atomic ResponsePlan activation (Stage 1 of the Global Multi-Incident Optimizer refactor).

ResponsePlanActivationService is the one place a candidate, already-
persisted ResponsePlan (see ResponseOptimizationAgent.optimize_from_input,
which already saves it in its own, separate, already-committed
transaction - that separation is a pre-existing, deliberate architectural
choice this service preserves, not something it changes) is turned into
the FireEvent's CURRENT plan. It atomically:
  (a) locks the FireEvent row;
  (b) locks every selected FirefightingResource row, in deterministic
      resource_id order;
  (c) verifies each resource still exists, is operationally eligible, and
      is not committed to a DIFFERENT FireEvent;
  (d) releases this FireEvent's own prior commitments and creates new
      ones for the candidate plan;
  (e) persists the ResponsePlanPlanningState sidecar that
      CurrentResponsePlanResolver already treats as "this plan is
      current";
all in ONE transaction, committed or rolled back together. If activation
fails for any reason, the previous current plan and its commitments are
left exactly as they were.

FirefightingResource.status (AVAILABLE/ASSIGNED/UNAVAILABLE) and
ResourceCommitment ownership are deliberately kept separate concepts: a
resource already committed to THIS FireEvent remains eligible for this
FireEvent's own replan regardless of its raw operational status (Task 12) -
only a resource with NO existing commitment must be AVAILABLE to be newly
claimed. This service never writes FirefightingResource.status itself.

Written to activate one FireEvent's plan at a time (Stage 1's scope), but
the persistence it writes - individual ResourceCommitment rows keyed by
resource_id, each carrying its own fire_event_id - places no constraint
that would prevent a future global activation from writing commitment rows
for several different FireEvents inside one transaction (Stage 2+).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.fire_event_db import FireEventDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.models.fire_event_status import FireEventStatus
from src.models.resource_status import ResourceStatus
from src.repositories.response_plan_planning_state_repository import StoredResponsePlanPlanningState
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.services.resource_reservation.resource_commitment_conflict import ResourceCommitmentConflict

_ACTIVE_STATUSES = frozenset({FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED})


class ResponsePlanActivationService:
    """Atomically activate a candidate ResponsePlan: commitments + current-plan sidecar together."""

    def __init__(
        self,
        response_plan_repository: ResponsePlanRepository | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()
        self._session_factory = session_factory or get_session_factory()

    def activate(
        self,
        *,
        response_plan_id: int,
        planning_effective_state_fingerprint: str,
        as_of: datetime,
    ) -> StoredResponsePlanPlanningState:
        """Make `response_plan_id` the FireEvent's current plan, atomically.

        Raises ResourceCommitmentConflict if any selected resource could not
        be acquired (already committed elsewhere, no longer AVAILABLE, or no
        longer exists) - in that case nothing is written: the previous
        current plan and its commitments are left exactly as they were.
        """
        self._validate_request(response_plan_id, planning_effective_state_fingerprint, as_of)

        stored_plan = self._response_plan_repository.get_by_id(response_plan_id)
        if stored_plan is None:
            raise ValueError(f"ResponsePlan {response_plan_id!r} was not found.")
        fire_event_id = stored_plan.plan.fire_event_id
        requested_resource_ids = tuple(sorted({action.resource_id for action in stored_plan.plan.actions}))

        session = self._session_factory()
        try:
            # 1. Lock the FireEvent row - serializes competing activations for the SAME event (Task 21).
            db_event = session.execute(
                select(FireEventDB).where(FireEventDB.id == fire_event_id).with_for_update()
            ).scalar_one_or_none()
            if db_event is None:
                raise ValueError(f"FireEvent {fire_event_id!r} was not found.")
            if FireEventStatus(db_event.status) not in _ACTIVE_STATUSES:
                raise ValueError(f"FireEvent {fire_event_id!r} is no longer active; cannot activate a plan for it.")

            # 2. Lock every selected resource row, in deterministic sorted order (Task 8).
            db_resources = ()
            if requested_resource_ids:
                db_resources = (
                    session.execute(
                        select(FirefightingResourceDB)
                        .where(FirefightingResourceDB.id.in_(requested_resource_ids))
                        .order_by(FirefightingResourceDB.id)
                        .with_for_update()
                    )
                    .scalars()
                    .all()
                )

            # 3 & 4. Re-read operational status and commitment ownership (now safely, under the resource-row lock).
            conflicts = self._find_conflicts(session, fire_event_id, requested_resource_ids, db_resources)
            if conflicts:
                session.rollback()
                raise ResourceCommitmentConflict(fire_event_id=fire_event_id, conflicts=conflicts)

            # 5 & 6. Release this event's old commitments, create new ones.
            self._resource_commitment_repository.replace_commitments_for_plan(
                session,
                fire_event_id=fire_event_id,
                response_plan_id=response_plan_id,
                resource_ids=requested_resource_ids,
                committed_at=as_of,
            )

            # 7. Persist the sidecar that makes this plan current.
            db_sidecar = ResponsePlanPlanningStateDB(
                response_plan_id=response_plan_id,
                planning_effective_state_fingerprint=planning_effective_state_fingerprint,
            )
            session.add(db_sidecar)
            session.flush()

            # 8. Commit everything together.
            session.commit()

            return StoredResponsePlanPlanningState(
                id=db_sidecar.id,
                response_plan_id=response_plan_id,
                planning_effective_state_fingerprint=planning_effective_state_fingerprint,
                created_at=db_sidecar.created_at,
            )
        except IntegrityError as exc:
            # Final-invariant safety net (Task 8): the resource_id PK/unique
            # constraint on resource_commitments catches anything the row
            # locking above should already have prevented.
            session.rollback()
            raise ResourceCommitmentConflict(
                fire_event_id=fire_event_id,
                conflicts={resource_id: "concurrent_commitment_conflict" for resource_id in requested_resource_ids},
            ) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _find_conflicts(
        session: Session,
        fire_event_id: int,
        requested_resource_ids: tuple[str, ...],
        db_resources,
    ) -> dict[str, str]:
        conflicts: dict[str, str] = {}
        found_ids = {resource.id for resource in db_resources}
        for missing_id in sorted(set(requested_resource_ids) - found_ids):
            conflicts[missing_id] = "resource_not_found"

        if not requested_resource_ids:
            return conflicts

        existing_commitments = (
            session.execute(
                select(ResourceCommitmentDB).where(ResourceCommitmentDB.resource_id.in_(requested_resource_ids))
            )
            .scalars()
            .all()
        )
        commitment_by_resource_id = {commitment.resource_id: commitment for commitment in existing_commitments}

        for db_resource in db_resources:
            existing = commitment_by_resource_id.get(db_resource.id)
            owned_by_this_event = existing is not None and existing.fire_event_id == fire_event_id
            if existing is not None and existing.fire_event_id != fire_event_id:
                conflicts[db_resource.id] = f"committed_to_fire_event_{existing.fire_event_id}"
                continue
            if db_resource.status is not ResourceStatus.AVAILABLE and not owned_by_this_event:
                conflicts[db_resource.id] = "not_available"
        return conflicts

    @staticmethod
    def _validate_request(response_plan_id: object, fingerprint: object, as_of: object) -> None:
        if isinstance(response_plan_id, bool) or not isinstance(response_plan_id, int) or response_plan_id <= 0:
            raise ValueError(f"response_plan_id must be a positive integer, got {response_plan_id!r}")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(
                f"planning_effective_state_fingerprint must be a non-empty string, got {fingerprint!r}"
            )
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
