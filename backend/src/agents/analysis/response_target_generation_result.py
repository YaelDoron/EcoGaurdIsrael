"""Result object for a ResponseTargetGenerationAgent run."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.models.response_target import ResponseTarget


class ResponseTargetGenerationStatus(Enum):
    """Business/operational outcome of response-target generation."""

    GENERATED = "generated"
    INACTIVE_EVENT = "inactive_event"
    FAILED = "failed"


@dataclass(frozen=True)
class ResponseTargetGenerationResult:
    """Outcome of one response-target generation orchestration run."""

    success: bool
    fire_event_id: int
    status: ResponseTargetGenerationStatus
    target_set_id: int | None
    targets: tuple[ResponseTarget, ...] = ()
    target_count: int = 0
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError(f"success must be a bool, got {self.success!r}.")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}.")
        if not isinstance(self.status, ResponseTargetGenerationStatus):
            raise ValueError(f"status must be a ResponseTargetGenerationStatus, got {self.status!r}.")

        targets = tuple(self.targets)
        for target in targets:
            if not isinstance(target, ResponseTarget):
                raise ValueError(f"targets must contain ResponseTarget items, got {target!r}.")
            if target.fire_event_id != self.fire_event_id:
                raise ValueError("target fire_event_id must match result fire_event_id.")
        object.__setattr__(self, "targets", targets)

        if isinstance(self.target_count, bool) or not isinstance(self.target_count, int) or self.target_count < 0:
            raise ValueError(f"target_count must be a non-negative integer, got {self.target_count!r}.")
        if self.target_count != len(targets):
            raise ValueError("target_count must equal len(targets).")

        if self.status is ResponseTargetGenerationStatus.GENERATED:
            self._validate_generated_result(targets)
        elif self.status is ResponseTargetGenerationStatus.INACTIVE_EVENT:
            self._validate_inactive_result(targets)
        elif self.status is ResponseTargetGenerationStatus.FAILED:
            self._validate_failed_result(targets)

    def _validate_generated_result(self, targets: tuple[ResponseTarget, ...]) -> None:
        if not self.success:
            raise ValueError("GENERATED response-target results must be successful.")
        if isinstance(self.target_set_id, bool) or not isinstance(self.target_set_id, int) or self.target_set_id <= 0:
            raise ValueError("GENERATED response-target results must include target_set_id.")
        if not targets:
            raise ValueError("GENERATED response-target results must include targets.")
        if self.error_message is not None:
            raise ValueError("GENERATED response-target results must not include error_message.")

    def _validate_inactive_result(self, targets: tuple[ResponseTarget, ...]) -> None:
        if not self.success:
            raise ValueError("INACTIVE_EVENT response-target results must be successful.")
        if self.target_set_id is not None:
            raise ValueError("INACTIVE_EVENT response-target results must not include target_set_id.")
        if targets:
            raise ValueError("INACTIVE_EVENT response-target results must not include targets.")
        if self.error_message is not None:
            raise ValueError("INACTIVE_EVENT response-target results must not include error_message.")

    def _validate_failed_result(self, targets: tuple[ResponseTarget, ...]) -> None:
        if self.success:
            raise ValueError("FAILED response-target results must not be successful.")
        if self.target_set_id is not None:
            raise ValueError("FAILED response-target results must not include target_set_id.")
        if targets:
            raise ValueError("FAILED response-target results must not include targets.")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("FAILED response-target results must include error_message.")
