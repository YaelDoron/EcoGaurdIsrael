"""Global optimizer-only allocation-slot abstraction (Stage 4 of the Global
Multi-Incident Optimizer refactor, Task 3).

A GlobalAllocationSlot is the GA's actual unit of coverage - not a
ResponseTarget directly - so a future stage can generate several slots for
one demanding target (e.g. a CRITICAL fire needing several trucks) without
redesigning the resource-indexed chromosome. Stage 4 itself only ever
creates exactly one slot per ResponseTarget (`slot_index=0`,
`required=False` - a neutral placeholder with no severity semantics yet);
Stage 5 owns deciding how many slots a target actually needs.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import validate_non_negative_int, validate_non_empty_string, validate_positive_int


@dataclass(frozen=True)
class GlobalAllocationSlot:
    """One unit of coverage the GA can assign a resource to."""

    slot_id: str
    fire_event_id: int
    response_target_id: int
    slot_index: int
    required: bool

    def __post_init__(self) -> None:
        validate_non_empty_string("slot_id", self.slot_id)
        validate_positive_int("fire_event_id", self.fire_event_id)
        validate_positive_int("response_target_id", self.response_target_id)
        validate_non_negative_int("slot_index", self.slot_index)
        if not isinstance(self.required, bool):
            raise ValueError(f"required must be a bool, got {self.required!r}")
