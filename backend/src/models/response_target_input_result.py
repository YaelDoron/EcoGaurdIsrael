"""Prepared input for operational response-target generation."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.response_target_input import ResponseTargetInput
from src.models.response_target_input_status import ResponseTargetInputStatus


@dataclass(frozen=True)
class ResponseTargetInputResult:
    """Response-target input preparation result."""

    status: ResponseTargetInputStatus
    input_data: ResponseTargetInput | None
    fire_event_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResponseTargetInputStatus):
            raise ValueError(f"status must be a ResponseTargetInputStatus, got {self.status!r}")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")

        if self.status is ResponseTargetInputStatus.READY:
            if not isinstance(self.input_data, ResponseTargetInput):
                raise ValueError("READY response-target input result requires input_data.")
            if self.input_data.fire_event_id != self.fire_event_id:
                raise ValueError("input_data.fire_event_id must match result fire_event_id.")
        elif self.input_data is not None:
            raise ValueError(f"{self.status.value} response-target input result must not include input_data.")
