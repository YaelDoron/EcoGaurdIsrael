"""Domain exception for a failed atomic plan activation (Stage 1).

Raised by ResponsePlanActivationService.activate() instead of leaking a raw
SQLAlchemy IntegrityError - callers must be able to distinguish "the
candidate plan itself was invalid/optimization failed" from "a valid
optimization result lost a concurrency race, or one of its resources
stopped being operationally eligible between planning and activation."
Both surface as this one exception, since the caller's correct response is
identical in both cases (discard the candidate, rebuild context, retry) -
see response_planning_refresh_orchestrator.py's bounded retry (Task 16).
"""
from __future__ import annotations


class ResourceCommitmentConflict(Exception):
    """Activation could not acquire every requested resource for this FireEvent.

    `conflicts` maps each contested resource_id to a short, safe reason
    string - e.g. "committed_to_fire_event_7" or "not_available" - never a
    raw database exception message.
    """

    def __init__(self, *, fire_event_id: int, conflicts: dict[str, str]) -> None:
        self.fire_event_id = fire_event_id
        self.conflicts = dict(conflicts)
        conflict_summary = ", ".join(f"{resource_id} ({reason})" for resource_id, reason in sorted(self.conflicts.items()))
        super().__init__(f"FireEvent {fire_event_id} activation conflict: {conflict_summary}")
