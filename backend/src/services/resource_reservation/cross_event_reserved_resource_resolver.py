"""Stage-0/Stage-1 collision-prevention read helper (Global Multi-Incident Optimizer refactor).

CrossEventReservedResourceResolver answers exactly one question: "which
resource ids are already used by some OTHER active FireEvent?" It is a pure
read/derivation over already-persisted data - it never persists anything,
never changes FirefightingResource.status, and never invokes an agent.

As of Stage 1, the result is the UNION of two sources:

1. Explicit ResourceCommitment rows (ResourceCommitmentRepository) - the
   authoritative, transaction-safe source going forward: every plan
   activated through ResponsePlanActivationService writes these.
2. The Stage-0 inferred set (other active events' current ResponsePlan
   resource ids, via ResponsePlanRepository) - kept as a COMPATIBILITY
   FALLBACK, not because it is still authoritative, but because a current
   plan activated *before* Stage 1 shipped has no ResourceCommitment row at
   all (Stage 1's migration deliberately does not backfill historical
   current plans - see the Stage-1 migration script/report). Without this
   union, such a pre-Stage-1 current plan's resources would incorrectly
   appear unreserved to a new FireEvent being planned today.

   This fallback can be safely REMOVED once every FireEvent active at the
   time has gone through at least one activation under Stage 1 (i.e. every
   current plan has a commitment row) - in practice, once enough time/
   refresh cycles have passed after the Stage-1 deploy that no pre-Stage-1
   current plan is still current. There is no code-level trigger for this;
   it is an operational decision for whoever removes it.

Duplicates between the two sources are naturally deduplicated by returning
a set. Stage 0's own limitation (inferring reservation from persisted plans
rather than a real lock) no longer applies to source 1, which Stage 1's
atomic, row-locked activation makes transaction-safe - see
ResponsePlanActivationService.
"""
from __future__ import annotations

from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_repository import ResponsePlanRepository


class CrossEventReservedResourceResolver:
    """Read which resource ids are reserved by other active FireEvents (commitments + Stage-0 fallback)."""

    def __init__(
        self,
        fire_event_repository: FireEventRepository | None = None,
        response_plan_repository: ResponsePlanRepository | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()

    def get_resource_ids_reserved_by_other_active_plans(self, *, excluded_fire_event_id: int) -> frozenset[str]:
        """Return resource ids reserved by another active FireEvent.

        `excluded_fire_event_id` is the FireEvent currently being planned -
        its OWN commitments/current plan resources are never included in
        the result, so a FireEvent replanning itself always keeps its own
        already-assigned resources eligible.
        """
        self._validate_fire_event_id(excluded_fire_event_id)

        other_active_fire_event_ids = tuple(
            fire_event_id
            for fire_event_id in self._fire_event_repository.get_response_eligible_fire_event_ids()
            if fire_event_id != excluded_fire_event_id
        )
        if not other_active_fire_event_ids:
            return frozenset()

        committed_resource_ids = self._resource_commitment_repository.get_resource_ids_for_other_active_events(
            excluded_fire_event_id
        )
        inferred_resource_ids = self._response_plan_repository.get_current_plan_resource_ids_for_fire_events(
            other_active_fire_event_ids
        )

        return frozenset(committed_resource_ids | inferred_resource_ids)

    @staticmethod
    def _validate_fire_event_id(fire_event_id: object) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
