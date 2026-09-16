"""Result of building the current Planning Effective State for one FireEvent."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.planning_effective_state import PlanningEffectiveState
from src.models.planning_effective_state_status import PlanningEffectiveStateStatus


@dataclass(frozen=True)
class PlanningEffectiveStateResult:
    """Outcome of one PlanningEffectiveStateBuilder.build() call."""

    status: PlanningEffectiveStateStatus
    state: PlanningEffectiveState | None
    fire_event_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlanningEffectiveStateStatus):
            raise ValueError(f"status must be a PlanningEffectiveStateStatus, got {self.status!r}")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")

        if self.status is PlanningEffectiveStateStatus.BUILT:
            if not isinstance(self.state, PlanningEffectiveState):
                raise ValueError("BUILT planning effective state result requires state.")
            if self.state.fire_event_id != self.fire_event_id:
                raise ValueError("state.fire_event_id must match result fire_event_id.")
        elif self.state is not None:
            raise ValueError(f"{self.status.value} planning effective state result must not include state.")
