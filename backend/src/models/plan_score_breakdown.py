"""Pure score result for response-plan candidate evaluation."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import (
    validate_finite_non_negative_number,
    validate_non_negative_int,
    validate_positive_int,
)
from src.models.response_plan_status import ResponsePlanStatus


@dataclass(frozen=True)
class PlanScoreBreakdown:
    """Deterministic explanation of one response-action candidate score."""

    total_score: float
    raw_fitness: float
    coverage_score: float
    average_eta_seconds: float | None
    total_priority: float
    covered_priority: float
    covered_target_count: int
    total_target_count: int
    uncovered_target_ids: tuple[int, ...]
    status: ResponsePlanStatus

    def __post_init__(self) -> None:
        validate_finite_non_negative_number("total_score", self.total_score)
        validate_finite_non_negative_number("raw_fitness", self.raw_fitness)
        validate_finite_non_negative_number("coverage_score", self.coverage_score)
        validate_finite_non_negative_number("total_priority", self.total_priority)
        validate_finite_non_negative_number("covered_priority", self.covered_priority)
        if self.average_eta_seconds is not None:
            validate_finite_non_negative_number("average_eta_seconds", self.average_eta_seconds)
        validate_non_negative_int("covered_target_count", self.covered_target_count)
        validate_non_negative_int("total_target_count", self.total_target_count)
        if self.covered_target_count > self.total_target_count:
            raise ValueError("covered_target_count must not exceed total_target_count.")
        if self.total_score > 100:
            raise ValueError(f"total_score must be within [0, 100], got {self.total_score!r}")
        if self.coverage_score > 100:
            raise ValueError(f"coverage_score must be within [0, 100], got {self.coverage_score!r}")
        try:
            uncovered_ids = tuple(self.uncovered_target_ids)
        except TypeError as exc:
            raise ValueError("uncovered_target_ids must be iterable.") from exc
        for target_id in uncovered_ids:
            validate_positive_int("uncovered_target_id", target_id)
        if len(uncovered_ids) != len(set(uncovered_ids)):
            raise ValueError("uncovered_target_ids must be unique.")
        if not isinstance(self.status, ResponsePlanStatus):
            raise ValueError(f"status must be a ResponsePlanStatus, got {self.status!r}")
        object.__setattr__(self, "uncovered_target_ids", uncovered_ids)


def derive_response_plan_status(
    *,
    total_target_count: int,
    covered_target_count: int,
) -> ResponsePlanStatus:
    """Derive plan completeness from target coverage counts."""
    validate_non_negative_int("total_target_count", total_target_count)
    validate_non_negative_int("covered_target_count", covered_target_count)
    if covered_target_count > total_target_count:
        raise ValueError("covered_target_count must not exceed total_target_count.")
    if total_target_count > 0 and covered_target_count == total_target_count:
        return ResponsePlanStatus.COMPLETE
    if covered_target_count > 0:
        return ResponsePlanStatus.PARTIAL
    return ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS
