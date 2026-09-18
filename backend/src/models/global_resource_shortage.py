"""Global (all-incidents) resource demand-vs-supply summary (Stage 5 of the
Global Multi-Incident Optimizer refactor, Tasks 18/20/21;
locked_resources_preserved added Stage 6).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import validate_non_negative_int


@dataclass(frozen=True)
class GlobalResourceShortage:
    """Aggregate demand-vs-supply picture across every active FireEvent in one global optimization."""

    total_required: int
    total_desired: int
    total_assigned: int
    unmet_required: int
    unmet_desired: int
    candidate_assignable_resource_count: int
    committed_resource_count: int
    unavailable_resource_count: int
    locked_resources_preserved: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "total_required",
            "total_desired",
            "total_assigned",
            "unmet_required",
            "unmet_desired",
            "candidate_assignable_resource_count",
            "committed_resource_count",
            "unavailable_resource_count",
            "locked_resources_preserved",
        ):
            validate_non_negative_int(field_name, getattr(self, field_name))

        if self.total_desired < self.total_required:
            raise ValueError(
                f"total_desired ({self.total_desired!r}) must be >= total_required ({self.total_required!r})."
            )
        # Stage 6: total_assigned may exceed total_desired by up to
        # locked_resources_preserved - already-hard-dispatched resources
        # kept on an event beyond what a DECREASED desired demand would now
        # call for (Task 8: never fabricate an implicit recall) are real,
        # counted supply, not a violation of the severity-driven envelope.
        if self.total_assigned > self.total_desired + self.locked_resources_preserved:
            raise ValueError(
                f"total_assigned ({self.total_assigned!r}) must not exceed total_desired "
                f"({self.total_desired!r}) + locked_resources_preserved ({self.locked_resources_preserved!r})."
            )
        if self.unmet_desired < self.unmet_required:
            raise ValueError(
                f"unmet_desired ({self.unmet_desired!r}) must be >= unmet_required ({self.unmet_required!r})."
            )

    @property
    def has_shortage(self) -> bool:
        return self.unmet_required > 0 or self.unmet_desired > 0

    @property
    def insufficient_supply(self) -> bool:
        """True when the candidate resource universe itself cannot satisfy desired demand -
        distinct from a slot going uncovered merely because no feasible ROUTE existed."""
        return self.candidate_assignable_resource_count < self.total_desired
