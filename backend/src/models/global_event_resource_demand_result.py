"""Per-FireEvent demand-vs-supply outcome (Stage 5 of the Global
Multi-Incident Optimizer refactor, Tasks 18/19; locked_resources_preserved
added Stage 6). Lets the future UI show, e.g., "Fire B: minimum required =
2, desired = 3, assigned = 1, unmet minimum = 1, unmet desired = 2" without
reconstructing demand from slots.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import validate_non_negative_int, validate_positive_int


@dataclass(frozen=True)
class GlobalEventResourceDemandResult:
    """One FireEvent's configured demand alongside what the global optimization actually assigned it."""

    fire_event_id: int
    minimum_resources: int
    desired_resources: int
    suppression_resources_assigned: int
    required_slots_covered: int
    required_slots_uncovered: int
    desired_slots_covered: int
    desired_slots_uncovered: int
    predicted_risk_slots_covered: int
    locked_resources_preserved: int = 0

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        for field_name in (
            "minimum_resources",
            "desired_resources",
            "suppression_resources_assigned",
            "required_slots_covered",
            "required_slots_uncovered",
            "desired_slots_covered",
            "desired_slots_uncovered",
            "predicted_risk_slots_covered",
            "locked_resources_preserved",
        ):
            validate_non_negative_int(field_name, getattr(self, field_name))

        if self.desired_resources < self.minimum_resources:
            raise ValueError(
                f"desired_resources ({self.desired_resources!r}) must be >= minimum_resources ({self.minimum_resources!r})."
            )
        if self.required_slots_covered + self.required_slots_uncovered != self.minimum_resources:
            raise ValueError(
                "required_slots_covered + required_slots_uncovered must equal minimum_resources, got "
                f"{self.required_slots_covered!r} + {self.required_slots_uncovered!r} != {self.minimum_resources!r}"
            )
        if self.desired_slots_covered + self.desired_slots_uncovered != self.desired_resources - self.minimum_resources:
            raise ValueError(
                "desired_slots_covered + desired_slots_uncovered must equal (desired_resources - minimum_resources), "
                f"got {self.desired_slots_covered!r} + {self.desired_slots_uncovered!r} != "
                f"{self.desired_resources - self.minimum_resources!r}"
            )
        # Stage 6: locked_resources_preserved counts already-hard-dispatched
        # resources kept on this event BEYOND what required+desired slots
        # would otherwise call for (see GlobalAllocationSlotFactory's
        # locked-overflow slots) - a demand DECREASE must never fabricate an
        # implicit recall, so these are reported separately from severity-
        # driven required/desired accounting, never folded into it.
        if self.suppression_resources_assigned != (
            self.required_slots_covered + self.desired_slots_covered + self.locked_resources_preserved
        ):
            raise ValueError(
                "suppression_resources_assigned must equal required_slots_covered + desired_slots_covered + "
                f"locked_resources_preserved, got {self.suppression_resources_assigned!r} != "
                f"{self.required_slots_covered + self.desired_slots_covered + self.locked_resources_preserved!r}"
            )

    @property
    def unmet_minimum(self) -> int:
        return self.required_slots_uncovered

    @property
    def unmet_desired(self) -> int:
        return self.required_slots_uncovered + self.desired_slots_uncovered
