"""Global GA output (Stage 4 of the Global Multi-Incident Optimizer
refactor, Task 21). Pure result data - no persistence happens here or
because of this object existing (Task 38: Stage 4 never saves anything).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.calculators.global_response_optimization.global_assignment_change_calculator import GlobalAssignmentChange
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.models.global_event_optimization_result import GlobalEventOptimizationResult
from src.models.global_resource_shortage import GlobalResourceShortage
from src.models.global_response_action import GlobalResponseAction
from src.models.optimization_validation import validate_non_empty_string, validate_positive_int


@dataclass(frozen=True)
class GlobalOptimizationResult:
    """The one global allocation produced by GlobalResponseOptimizationService.optimize()."""

    global_planning_run_id: int
    input_fingerprint: str
    optimization_methodology: str
    optimization_methodology_version: str
    random_seed: int
    fitness_score: float
    coverage_score: float
    average_eta_seconds: float | None
    actions: tuple[GlobalResponseAction, ...]
    uncovered_slot_ids: tuple[str, ...]
    event_results: tuple[GlobalEventOptimizationResult, ...]
    shortage: GlobalResourceShortage
    config: GlobalResponseOptimizationConfig
    assignment_changes: tuple[GlobalAssignmentChange, ...] = ()

    def __post_init__(self) -> None:
        validate_positive_int("global_planning_run_id", self.global_planning_run_id)
        validate_non_empty_string("input_fingerprint", self.input_fingerprint)
        validate_non_empty_string("optimization_methodology", self.optimization_methodology)
        validate_non_empty_string("optimization_methodology_version", self.optimization_methodology_version)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError(f"random_seed must be an integer, got {self.random_seed!r}")
        # fitness_score is Stage 5's tiered, deliberately UNBOUNDED ranking
        # value (see GlobalDemandScoringPolicy) - only coverage_score remains
        # a true 0-100 percentage.
        self._validate_finite("fitness_score", self.fitness_score)
        self._validate_score("coverage_score", self.coverage_score)
        if self.average_eta_seconds is not None:
            self._validate_non_negative_finite("average_eta_seconds", self.average_eta_seconds)

        actions = tuple(self.actions)
        for action in actions:
            if not isinstance(action, GlobalResponseAction):
                raise ValueError(f"actions must contain GlobalResponseAction items, got {action!r}")
        resource_ids = [action.resource_id for action in actions]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("actions must not assign the same resource_id more than once.")
        object.__setattr__(self, "actions", actions)

        uncovered_slot_ids = tuple(self.uncovered_slot_ids)
        for slot_id in uncovered_slot_ids:
            validate_non_empty_string("uncovered_slot_ids", slot_id)
        object.__setattr__(self, "uncovered_slot_ids", uncovered_slot_ids)

        event_results = tuple(self.event_results)
        for event_result in event_results:
            if not isinstance(event_result, GlobalEventOptimizationResult):
                raise ValueError(
                    f"event_results must contain GlobalEventOptimizationResult items, got {event_result!r}"
                )
        fire_event_ids = [event_result.fire_event_id for event_result in event_results]
        if len(set(fire_event_ids)) != len(fire_event_ids):
            raise ValueError("event_results must not contain duplicate fire_event_id entries.")
        object.__setattr__(self, "event_results", event_results)

        if not isinstance(self.shortage, GlobalResourceShortage):
            raise ValueError(f"shortage must be a GlobalResourceShortage, got {self.shortage!r}")

        if not isinstance(self.config, GlobalResponseOptimizationConfig):
            raise ValueError(f"config must be a GlobalResponseOptimizationConfig, got {self.config!r}")

        assignment_changes = tuple(self.assignment_changes)
        for change in assignment_changes:
            if not isinstance(change, GlobalAssignmentChange):
                raise ValueError(f"assignment_changes must contain GlobalAssignmentChange items, got {change!r}")
        change_resource_ids = [change.resource_id for change in assignment_changes]
        if len(set(change_resource_ids)) != len(change_resource_ids):
            raise ValueError("assignment_changes must not contain duplicate resource_id entries.")
        object.__setattr__(self, "assignment_changes", assignment_changes)

    @staticmethod
    def _validate_finite(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")

    @staticmethod
    def _validate_score(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{field_name} must be within [0, 100], got {value!r}")

    @staticmethod
    def _validate_non_negative_finite(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative finite number, got {value!r}")
