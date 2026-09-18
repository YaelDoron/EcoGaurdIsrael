"""CurrentGlobalAssignment: pre-GA-loaded snapshot of one resource's
existing operational assignment (Stage 6 of the Global Multi-Incident
Optimizer refactor, Task 14).

ResourceCommitment alone gives resource_id/fire_event_id/response_plan_id/
dispatch_state. Stability, audit, and assignment-change reporting also need
which exact response_target_id (slot) the resource is currently satisfying,
which ResourceCommitment does not carry - so this model derives it from the
current ResponsePlan's actions and folds in the commitment's dispatch_state.

Loaded ONCE, outside the Global GA, by a dedicated (DB-touching) loader
service and passed into GlobalPlanningInput as plain, immutable data - the
GA package itself never queries a repository (Task 14/60).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.dispatch_state import DispatchState
from src.models.optimization_validation import validate_non_empty_string, validate_positive_int


@dataclass(frozen=True)
class CurrentGlobalAssignment:
    """One resource's current commitment, resolved down to the exact target it serves."""

    resource_id: str
    fire_event_id: int
    response_target_id: int
    response_plan_id: int
    dispatch_state: DispatchState

    def __post_init__(self) -> None:
        validate_non_empty_string("resource_id", self.resource_id)
        validate_positive_int("fire_event_id", self.fire_event_id)
        validate_positive_int("response_target_id", self.response_target_id)
        validate_positive_int("response_plan_id", self.response_plan_id)
        if not isinstance(self.dispatch_state, DispatchState):
            raise ValueError(f"dispatch_state must be a DispatchState, got {self.dispatch_state!r}")
