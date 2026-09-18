"""Per-FireEvent projection of one GlobalOptimizationResult (Stage 4 of the
Global Multi-Incident Optimizer refactor, Task 22).

This is a GROUPING of the one global answer by fire_event_id - never a
second optimization. No GA runs to produce this; it is pure filtering +
aggregation over the already-decided global actions/slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.global_event_resource_demand_result import GlobalEventResourceDemandResult
from src.models.global_response_action import GlobalResponseAction
from src.models.optimization_validation import validate_positive_int


@dataclass(frozen=True)
class GlobalEventOptimizationResult:
    """One FireEvent's slice of the global optimization result."""

    fire_event_id: int
    actions: tuple[GlobalResponseAction, ...]
    covered_slot_ids: tuple[str, ...]
    uncovered_slot_ids: tuple[str, ...]
    coverage_score: float
    average_eta_seconds: float | None
    demand_result: GlobalEventResourceDemandResult

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        if not isinstance(self.demand_result, GlobalEventResourceDemandResult):
            raise ValueError(f"demand_result must be a GlobalEventResourceDemandResult, got {self.demand_result!r}")
        if self.demand_result.fire_event_id != self.fire_event_id:
            raise ValueError(
                f"demand_result.fire_event_id {self.demand_result.fire_event_id!r} does not match this "
                f"projection's fire_event_id {self.fire_event_id!r}."
            )

        actions = tuple(self.actions)
        for action in actions:
            if not isinstance(action, GlobalResponseAction):
                raise ValueError(f"actions must contain GlobalResponseAction items, got {action!r}")
            if action.fire_event_id != self.fire_event_id:
                raise ValueError(
                    f"action fire_event_id {action.fire_event_id!r} does not match this "
                    f"projection's fire_event_id {self.fire_event_id!r}."
                )
        object.__setattr__(self, "actions", actions)

        covered_slot_ids = tuple(self.covered_slot_ids)
        uncovered_slot_ids = tuple(self.uncovered_slot_ids)
        if set(covered_slot_ids) & set(uncovered_slot_ids):
            raise ValueError("covered_slot_ids and uncovered_slot_ids must not overlap.")
        object.__setattr__(self, "covered_slot_ids", covered_slot_ids)
        object.__setattr__(self, "uncovered_slot_ids", uncovered_slot_ids)

        self._validate_score("coverage_score", self.coverage_score)
        if self.average_eta_seconds is not None:
            self._validate_non_negative_finite("average_eta_seconds", self.average_eta_seconds)

    @staticmethod
    def _validate_score(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{field_name} must be within [0, 100], got {value!r}")

    @staticmethod
    def _validate_non_negative_finite(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative finite number, got {value!r}")
