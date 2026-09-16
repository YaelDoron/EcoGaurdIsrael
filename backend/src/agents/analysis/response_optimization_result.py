"""Result object for a ResponseOptimizationAgent run."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.models.response_plan_status import ResponsePlanStatus


class ResponseOptimizationStatus(Enum):
    """Business/technical outcome of response-plan optimization."""

    OPTIMIZED = "optimized"
    FAILED = "failed"


@dataclass(frozen=True)
class ResponseOptimizationResult:
    """Outcome of one response optimization orchestration run."""

    success: bool
    fire_event_id: int
    response_target_set_id: int | None
    route_planning_run_id: int | None
    status: ResponseOptimizationStatus
    plan_status: ResponsePlanStatus | None
    response_plan_id: int | None
    action_count: int = 0
    uncovered_target_count: int = 0
    plan_score: float | None = None
    coverage_score: float | None = None
    average_eta_seconds: float | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError(f"success must be a bool, got {self.success!r}.")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}.")
        if not isinstance(self.status, ResponseOptimizationStatus):
            raise ValueError(f"status must be a ResponseOptimizationStatus, got {self.status!r}.")
        if isinstance(self.action_count, bool) or not isinstance(self.action_count, int) or self.action_count < 0:
            raise ValueError(f"action_count must be a non-negative integer, got {self.action_count!r}.")
        if (
            isinstance(self.uncovered_target_count, bool)
            or not isinstance(self.uncovered_target_count, int)
            or self.uncovered_target_count < 0
        ):
            raise ValueError(
                f"uncovered_target_count must be a non-negative integer, got {self.uncovered_target_count!r}."
            )

        if self.status is ResponseOptimizationStatus.OPTIMIZED:
            self._validate_optimized_result()
        elif self.status is ResponseOptimizationStatus.FAILED:
            self._validate_failed_result()

    def _validate_optimized_result(self) -> None:
        if not self.success:
            raise ValueError("OPTIMIZED response-optimization results must be successful.")
        for field_name, value in (
            ("response_target_set_id", self.response_target_set_id),
            ("route_planning_run_id", self.route_planning_run_id),
            ("response_plan_id", self.response_plan_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"OPTIMIZED results must include positive {field_name}.")
        if not isinstance(self.plan_status, ResponsePlanStatus):
            raise ValueError("OPTIMIZED results must include plan_status.")
        if self.error_message is not None:
            raise ValueError("OPTIMIZED results must not include error_message.")

    def _validate_failed_result(self) -> None:
        if self.success:
            raise ValueError("FAILED response-optimization results must not be successful.")
        if self.response_plan_id is not None:
            raise ValueError("FAILED response-optimization results must not include response_plan_id.")
        if self.plan_status is not None:
            raise ValueError("FAILED response-optimization results must not include plan_status.")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("FAILED response-optimization results must include error_message.")
