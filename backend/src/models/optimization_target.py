"""Pure response-optimization target input model."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.optimization_validation import (
    validate_finite_non_negative_number,
    validate_non_negative_int,
    validate_positive_int,
)
from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class OptimizationTarget:
    """Persisted response target normalized for assignment optimization."""

    response_target_id: int
    target_order: int
    target_type: ResponseTargetType
    priority_score: float

    def __post_init__(self) -> None:
        validate_positive_int("response_target_id", self.response_target_id)
        validate_non_negative_int("target_order", self.target_order)
        if not isinstance(self.target_type, ResponseTargetType):
            raise ValueError(f"target_type must be a ResponseTargetType, got {self.target_type!r}")
        validate_finite_non_negative_number("priority_score", self.priority_score)

    @property
    def ordering_key(self) -> tuple[int, int]:
        return (self.target_order, self.response_target_id)
