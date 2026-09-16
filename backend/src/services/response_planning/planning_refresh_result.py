"""Top-level result contracts for ResponsePlanningRefreshOrchestrator (Epic 5, US 5.4, Task 5)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlanningRefreshStatus(Enum):
    """Business/operational outcome of one US 5.4 planning-refresh request."""

    REFRESHED = "refreshed"
    NO_OP = "no_op"
    INACTIVE_EVENT = "inactive_event"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


@dataclass(frozen=True)
class PlanningRefreshResult:
    """Outcome of one ResponsePlanningRefreshOrchestrator.refresh() call."""

    status: PlanningRefreshStatus
    fire_event_id: int
    route_planning_run_id: int | None
    response_plan_id: int | None
    comparison_id: int | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PlanningRefreshStatus):
            raise ValueError(f"status must be a PlanningRefreshStatus, got {self.status!r}")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        self._validate_optional_positive_int("route_planning_run_id", self.route_planning_run_id)
        self._validate_optional_positive_int("response_plan_id", self.response_plan_id)
        self._validate_optional_positive_int("comparison_id", self.comparison_id)

        if self.status is PlanningRefreshStatus.FAILED:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("FAILED planning-refresh results must include a non-empty error.")
        elif self.error is not None:
            raise ValueError(f"{self.status.value} planning-refresh results must not include error.")

        if self.status in (PlanningRefreshStatus.NO_OP, PlanningRefreshStatus.REFRESHED):
            self._require_all_ids_present()
        elif self.status in (PlanningRefreshStatus.INACTIVE_EVENT, PlanningRefreshStatus.INSUFFICIENT_DATA):
            self._require_no_ids_present()

    def _require_all_ids_present(self) -> None:
        if self.route_planning_run_id is None or self.response_plan_id is None or self.comparison_id is None:
            raise ValueError(
                f"{self.status.value} planning-refresh results must include route_planning_run_id, "
                "response_plan_id, and comparison_id."
            )

    def _require_no_ids_present(self) -> None:
        if (
            self.route_planning_run_id is not None
            or self.response_plan_id is not None
            or self.comparison_id is not None
        ):
            raise ValueError(
                f"{self.status.value} planning-refresh results must not include route_planning_run_id, "
                "response_plan_id, or comparison_id."
            )

    @staticmethod
    def _validate_optional_positive_int(field_name: str, value: object) -> None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{field_name} must be a positive integer or None, got {value!r}")
