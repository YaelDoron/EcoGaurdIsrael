"""Result object for a FireDetectionAgent run."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FireDetectionResult:
    """Outcome of one active wildfire detection orchestration run."""

    success: bool
    candidates_processed: int
    no_event_count: int
    events_created: int
    events_updated: int
    event_ids: tuple[int, ...]
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError(f"success must be a bool, got {self.success!r}.")
        for field_name in ("candidates_processed", "no_event_count", "events_created", "events_updated"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}.")
        event_ids = tuple(sorted(tuple(self.event_ids)))
        if any(isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0 for event_id in event_ids):
            raise ValueError(f"event_ids must contain positive integer ids, got {event_ids!r}.")
        object.__setattr__(self, "event_ids", event_ids)

        if self.success and self.error_message is not None:
            raise ValueError("successful fire-detection results must not include error_message.")
        if not self.success and (not isinstance(self.error_message, str) or not self.error_message.strip()):
            raise ValueError("failed fire-detection results must include error_message.")
