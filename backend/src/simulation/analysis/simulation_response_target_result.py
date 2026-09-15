"""Result for simulation-triggered response-target generation."""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.analysis.response_target_generation_result import (
    ResponseTargetGenerationResult,
    ResponseTargetGenerationStatus,
)


@dataclass(frozen=True)
class SimulationResponseTargetResult:
    """Whether a simulation event triggered response-target generation."""

    triggered: bool
    generation_results: tuple[ResponseTargetGenerationResult, ...] = ()
    fire_event_ids: tuple[int, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool):
            raise ValueError(f"triggered must be a bool, got {self.triggered!r}")

        generation_results = tuple(self.generation_results)
        for generation_result in generation_results:
            if not isinstance(generation_result, ResponseTargetGenerationResult):
                raise ValueError(
                    "generation_results must contain ResponseTargetGenerationResult items, "
                    f"got {generation_result!r}"
                )
        object.__setattr__(self, "generation_results", generation_results)

        fire_event_ids = tuple(sorted(set(self.fire_event_ids)))
        for fire_event_id in fire_event_ids:
            if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
                raise ValueError(f"fire_event_ids must contain positive integer ids, got {fire_event_id!r}")
        object.__setattr__(self, "fire_event_ids", fire_event_ids)

        if self.triggered:
            if self.reason is not None:
                raise ValueError("triggered simulation response-target results must not include reason.")
            if not generation_results:
                raise ValueError("triggered simulation response-target results must include generation results.")
            if fire_event_ids != tuple(sorted({result.fire_event_id for result in generation_results})):
                raise ValueError("fire_event_ids must match the ids in generation_results.")
        else:
            if generation_results:
                raise ValueError("non-triggered simulation response-target results must not include generations.")
            if fire_event_ids:
                raise ValueError("non-triggered simulation response-target results must not include fire_event_ids.")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("non-triggered simulation response-target results must include reason.")

    @property
    def fire_events_requested(self) -> int:
        return len(self.fire_event_ids)

    @property
    def fire_events_processed(self) -> int:
        return len(self.generation_results)

    @property
    def target_sets_generated(self) -> int:
        return sum(1 for result in self.generation_results if result.status is ResponseTargetGenerationStatus.GENERATED)

    @property
    def inactive_events(self) -> int:
        return sum(
            1 for result in self.generation_results if result.status is ResponseTargetGenerationStatus.INACTIVE_EVENT
        )

    @property
    def failed_events(self) -> int:
        return sum(1 for result in self.generation_results if result.status is ResponseTargetGenerationStatus.FAILED)

    @property
    def target_count(self) -> int:
        return sum(result.target_count for result in self.generation_results)

    @property
    def target_set_ids(self) -> tuple[int, ...]:
        return tuple(result.target_set_id for result in self.generation_results if result.target_set_id is not None)

    @property
    def error_messages(self) -> tuple[str, ...]:
        return tuple(result.error_message for result in self.generation_results if result.error_message)
