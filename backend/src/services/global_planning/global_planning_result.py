"""Result contract for GlobalPlanningOrchestrator.run() (Stage 2 of the
Global Multi-Incident Optimizer refactor, Task 20). Exposes plain data -
never ORM objects - for demo output, monitoring, and future global-response
UI/audit consumers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus


@dataclass(frozen=True)
class GlobalPlanningEventResult:
    """One FireEvent's outcome within a GlobalPlanningResult."""

    fire_event_id: int
    planning_status: GlobalPlanningRunEventStatus
    response_plan_id: int | None

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if not isinstance(self.planning_status, GlobalPlanningRunEventStatus):
            raise ValueError(f"planning_status must be a GlobalPlanningRunEventStatus, got {self.planning_status!r}")
        if self.response_plan_id is not None and (
            isinstance(self.response_plan_id, bool)
            or not isinstance(self.response_plan_id, int)
            or self.response_plan_id <= 0
        ):
            raise ValueError(f"response_plan_id must be a positive integer or None, got {self.response_plan_id!r}")


@dataclass(frozen=True)
class GlobalPlanningResult:
    """Structured outcome of one GlobalPlanningOrchestrator.run() call."""

    global_planning_run_id: int
    status: GlobalPlanningRunStatus
    started_at: datetime
    completed_at: datetime | None
    input_fingerprint: str | None
    event_results: tuple[GlobalPlanningEventResult, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_planning_run_id, bool)
            or not isinstance(self.global_planning_run_id, int)
            or self.global_planning_run_id <= 0
        ):
            raise ValueError(
                f"global_planning_run_id must be a positive integer, got {self.global_planning_run_id!r}"
            )
        if not isinstance(self.status, GlobalPlanningRunStatus):
            raise ValueError(f"status must be a GlobalPlanningRunStatus, got {self.status!r}")
        if not isinstance(self.started_at, datetime) or self.started_at.tzinfo is None:
            raise ValueError(f"started_at must be a timezone-aware datetime, got {self.started_at!r}")
        if self.completed_at is not None and (
            not isinstance(self.completed_at, datetime) or self.completed_at.tzinfo is None
        ):
            raise ValueError(f"completed_at must be a timezone-aware datetime or None, got {self.completed_at!r}")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str) or not self.input_fingerprint.strip()
        ):
            raise ValueError(f"input_fingerprint must be a non-empty string or None, got {self.input_fingerprint!r}")

        event_results = tuple(self.event_results)
        for event_result in event_results:
            if not isinstance(event_result, GlobalPlanningEventResult):
                raise ValueError(f"event_results must contain GlobalPlanningEventResult items, got {event_result!r}")
        object.__setattr__(self, "event_results", event_results)
