"""Derived current/historical ResponsePlan read semantics (Epic 5, US 5.4, Task 8).

Distinguishes "latest persisted ResponsePlan" from "current planning-safe
ResponsePlan": a plan is only eligible to be considered current if it has a
persisted PlanningEffectiveState fingerprint sidecar (Task 4) - i.e. its
producing optimization run is known to have completed successfully.
PlanComparison is NOT required: baseline comparison is a separate, downstream
concern that Task 5 may still be retrying, and a plan missing only its
comparison is still a valid, current planning-safe recommendation.

This resolver is a pure READ/derivation concern. It never persists
is_current/superseded/current_plan_id, never updates or deletes a
ResponsePlan row, and never creates a sidecar. It does not replace or alter
Task 5's own refresh/NO_OP decision, which intentionally still keys off
ResponsePlanRepository.get_latest_for_fire_event() alone - see
response_planning_refresh_orchestrator.py. The two are deliberately
different questions:

- Task 5 ("must I replan?"): is the LATEST persisted plan sidecar-backed
  with a matching fingerprint? If not, always do a full new planning cycle -
  never fall back to an older plan's fingerprint.
- Task 8 ("what should be shown/used as current right now?"): scan
  newest-to-oldest for the newest plan that has a sidecar at all, regardless
  of fingerprint value.

For an inactive (RESOLVED/DISMISSED) FireEvent, resolve() still returns that
event's last planning-safe plan if one exists - it answers "what was the
last planning-safe recommendation for this event", not "is this event
currently active". Callers that need to know event activity status read
FireEventStatus separately (e.g. via FireEventRepository); this resolver
does not fold that concern in.
"""
from __future__ import annotations

from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository, StoredResponsePlan


class CurrentResponsePlanResolver:
    """Derive the current planning-safe ResponsePlan for a FireEvent from existing history."""

    def __init__(
        self,
        response_plan_repository: ResponsePlanRepository | None = None,
        response_plan_planning_state_repository: ResponsePlanPlanningStateRepository | None = None,
    ) -> None:
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._response_plan_planning_state_repository = (
            response_plan_planning_state_repository or ResponsePlanPlanningStateRepository()
        )

    def resolve(self, *, fire_event_id: int) -> StoredResponsePlan | None:
        """Return the newest ResponsePlan for this FireEvent that has a valid sidecar.

        ResponsePlanRepository.get_for_fire_event() already orders newest-first
        (generated_at desc, id desc - identical to get_latest_for_fire_event's
        own convention), so results are scanned in that order as-is; no
        re-sorting and no RoutePlanningRun/PlanComparison lookups are needed.
        """
        self._validate_fire_event_id(fire_event_id)
        for stored_plan in self._response_plan_repository.get_for_fire_event(fire_event_id):
            if self._response_plan_planning_state_repository.get_for_plan(stored_plan.id) is not None:
                return stored_plan
        return None

    @staticmethod
    def _validate_fire_event_id(fire_event_id: object) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
