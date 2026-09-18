"""Domain model for persistent firefighting-resource ownership (Stage 1 of the
Global Multi-Incident Optimizer refactor; dispatch_state added in Stage 6).

A ResourceCommitment means: this firefighting resource is currently
committed to the CURRENT ResponsePlan of this FireEvent. It is deliberately
NOT the same concept as FirefightingResource.status (AVAILABLE/ASSIGNED/
UNAVAILABLE, an operational concept) - see
src/services/resource_reservation/response_plan_activation_service.py for
why the two are kept separate.

This is current-state ownership, not historical assignment data - history
already exists via persisted ResponsePlan -> ResponseAction rows, so this
model carries only what is needed to answer "who owns this resource right
now": resource_id, fire_event_id, response_plan_id, committed_at, and
(Stage 6) dispatch_state - whether this ownership is still freely
reassignable (PLANNED) or a hard operational lock (DISPATCHED). See
src/models/dispatch_state.py for why this is the minimum necessary addition
rather than a larger resource state machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.models.dispatch_state import DispatchState


@dataclass(frozen=True)
class ResourceCommitment:
    """One firefighting resource's current commitment to one FireEvent's current plan."""

    resource_id: str
    fire_event_id: int
    response_plan_id: int
    committed_at: datetime
    dispatch_state: DispatchState = DispatchState.PLANNED

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError(f"resource_id must be a non-empty string, got {self.resource_id!r}")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if (
            isinstance(self.response_plan_id, bool)
            or not isinstance(self.response_plan_id, int)
            or self.response_plan_id <= 0
        ):
            raise ValueError(f"response_plan_id must be a positive integer, got {self.response_plan_id!r}")
        if not isinstance(self.committed_at, datetime) or self.committed_at.tzinfo is None:
            raise ValueError(f"committed_at must be a timezone-aware datetime, got {self.committed_at!r}")
        if not isinstance(self.dispatch_state, DispatchState):
            raise ValueError(f"dispatch_state must be a DispatchState, got {self.dispatch_state!r}")
