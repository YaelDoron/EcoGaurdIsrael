"""Pure domain model for a generated response-target snapshot."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.models.response_target import ResponseTarget
from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class ResponseTargetSet:
    """One ordered response-target generation result for a FireEvent."""

    fire_event_id: int
    generated_at: datetime
    methodology: str
    methodology_version: str
    targets: tuple[ResponseTarget, ...]

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if not isinstance(self.generated_at, datetime) or self.generated_at.tzinfo is None:
            raise ValueError(f"generated_at must be a timezone-aware datetime, got {self.generated_at!r}")
        self._validate_non_empty_string("methodology", self.methodology)
        self._validate_non_empty_string("methodology_version", self.methodology_version)

        try:
            targets = tuple(self.targets)
        except TypeError as exc:
            raise ValueError("targets must be iterable.") from exc
        if not targets:
            raise ValueError("ResponseTargetSet requires at least one target.")

        active_fire_count = 0
        for target in targets:
            if not isinstance(target, ResponseTarget):
                raise ValueError(f"targets must contain ResponseTarget items, got {target!r}")
            if target.fire_event_id != self.fire_event_id:
                raise ValueError(
                    "target fire_event_id must match target set fire_event_id, got "
                    f"{target.fire_event_id!r} for set {self.fire_event_id!r}"
                )
            if target.target_type is ResponseTargetType.ACTIVE_FIRE:
                active_fire_count += 1

        if active_fire_count != 1:
            raise ValueError(f"ResponseTargetSet requires exactly one ACTIVE_FIRE target, got {active_fire_count}.")

        object.__setattr__(self, "targets", targets)

    @staticmethod
    def _validate_non_empty_string(field_name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
