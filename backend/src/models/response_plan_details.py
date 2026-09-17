"""Read-only response-plan-details presentation DTOs for US 5.5.

These models are a pure, read-only presentation contract over data already
computed and persisted by US 5.1 (routing), US 5.2 (optimization), and
US 5.3 (baseline comparison). They perform no pathfinding, no optimization,
and no target-priority calculation of their own - assembling one of these
DTOs is nothing more than copying already-computed fields into a shape
convenient for operational display. Missing upstream data (no comparison
run yet, an uncovered target, a route with no path) is represented
explicitly with `None` / empty tuples rather than raising or guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

from src.models.optimization_validation import (
    validate_non_empty_string,
    validate_optional_finite_non_negative_number,
    validate_positive_int,
)
from src.models.response_target_type import ResponseTargetType


@dataclass(frozen=True)
class ResponseActionDetails:
    """One assigned resource within a response plan, enriched for display."""

    resource_id: str
    station_id: str
    response_target_id: int
    target_type: str
    target_priority: float
    eta_seconds: float | None
    route_distance_meters: float | None
    node_path: tuple[int, ...] | None

    def __post_init__(self) -> None:
        validate_non_empty_string("resource_id", self.resource_id)
        validate_non_empty_string("station_id", self.station_id)
        validate_positive_int("response_target_id", self.response_target_id)
        _validate_target_type(self.target_type)
        _validate_finite_number("target_priority", self.target_priority)
        validate_optional_finite_non_negative_number("eta_seconds", self.eta_seconds)
        validate_optional_finite_non_negative_number("route_distance_meters", self.route_distance_meters)

        if self.node_path is not None:
            node_path = _coerce_tuple("node_path", self.node_path)
            for node_id in node_path:
                validate_positive_int("node_path entry", node_id)
            object.__setattr__(self, "node_path", node_path)


@dataclass(frozen=True)
class OptimizationConfigDetails:
    """Presentation view of the exact GA configuration snapshot for one response plan (FND-06).

    A field-for-field display shape of the persisted `ResponseOptimizationConfig`
    (US 5.2/FND-04) - never the calculator type itself, so this module stays
    free of any dependency on `src.calculators`.
    """

    population_size: int
    generation_count: int
    mutation_rate: float
    crossover_rate: float
    eta_reference_seconds: float
    initial_assignment_probability: float
    tournament_size: int
    elitism_count: int

    def __post_init__(self) -> None:
        validate_positive_int("population_size", self.population_size)
        validate_positive_int("generation_count", self.generation_count)
        _validate_finite_number("mutation_rate", self.mutation_rate)
        _validate_finite_number("crossover_rate", self.crossover_rate)
        _validate_finite_number("eta_reference_seconds", self.eta_reference_seconds)
        _validate_finite_number("initial_assignment_probability", self.initial_assignment_probability)
        validate_positive_int("tournament_size", self.tournament_size)
        _validate_int("elitism_count", self.elitism_count)
        if self.elitism_count < 0:
            raise ValueError(f"elitism_count must be >= 0, got {self.elitism_count!r}")


@dataclass(frozen=True)
class BaselineComparisonDetails:
    """Presentation view of a persisted US 5.3 optimized-vs-baseline comparison."""

    baseline_score: float
    baseline_coverage_score: float
    baseline_average_eta_seconds: float | None
    score_difference: float
    improvement_percentage: float | None

    def __post_init__(self) -> None:
        _validate_finite_number("baseline_score", self.baseline_score)
        _validate_finite_number("baseline_coverage_score", self.baseline_coverage_score)
        validate_optional_finite_non_negative_number(
            "baseline_average_eta_seconds", self.baseline_average_eta_seconds
        )
        _validate_finite_number("score_difference", self.score_difference)
        if self.improvement_percentage is not None:
            _validate_finite_number("improvement_percentage", self.improvement_percentage)


@dataclass(frozen=True)
class ResponsePlanDetails:
    """Root read model for operational display of one response plan."""

    plan_id: int
    fire_event_id: int
    response_target_set_id: int
    route_planning_run_id: int
    generated_at: datetime
    methodology: str
    methodology_version: str
    random_seed: int
    is_current: bool
    plan_score: float
    coverage_score: float
    average_eta_seconds: float | None
    actions: tuple[ResponseActionDetails, ...]
    uncovered_target_ids: tuple[int, ...]
    baseline_comparison: BaselineComparisonDetails | None
    optimization_config: OptimizationConfigDetails | None

    def __post_init__(self) -> None:
        validate_positive_int("plan_id", self.plan_id)
        validate_positive_int("fire_event_id", self.fire_event_id)
        validate_positive_int("response_target_set_id", self.response_target_set_id)
        validate_positive_int("route_planning_run_id", self.route_planning_run_id)

        if not isinstance(self.generated_at, datetime) or self.generated_at.tzinfo is None:
            raise ValueError(f"generated_at must be a timezone-aware datetime, got {self.generated_at!r}")
        validate_non_empty_string("methodology", self.methodology)
        validate_non_empty_string("methodology_version", self.methodology_version)
        _validate_int("random_seed", self.random_seed)
        if not isinstance(self.is_current, bool):
            raise ValueError(f"is_current must be a bool, got {self.is_current!r}")

        _validate_finite_number("plan_score", self.plan_score)
        _validate_finite_number("coverage_score", self.coverage_score)
        validate_optional_finite_non_negative_number("average_eta_seconds", self.average_eta_seconds)

        actions = _coerce_tuple("actions", self.actions)
        for action in actions:
            if not isinstance(action, ResponseActionDetails):
                raise ValueError(f"actions must contain ResponseActionDetails items, got {action!r}")
        object.__setattr__(self, "actions", actions)

        uncovered_target_ids = _coerce_tuple("uncovered_target_ids", self.uncovered_target_ids)
        for target_id in uncovered_target_ids:
            validate_positive_int("uncovered_target_id", target_id)
        object.__setattr__(self, "uncovered_target_ids", uncovered_target_ids)

        if self.baseline_comparison is not None and not isinstance(
            self.baseline_comparison, BaselineComparisonDetails
        ):
            raise ValueError(
                f"baseline_comparison must be a BaselineComparisonDetails or None, got {self.baseline_comparison!r}"
            )

        if self.optimization_config is not None and not isinstance(
            self.optimization_config, OptimizationConfigDetails
        ):
            raise ValueError(
                "optimization_config must be an OptimizationConfigDetails or None, "
                f"got {self.optimization_config!r}"
            )


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc


def _validate_finite_number(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")


def _validate_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _validate_target_type(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"target_type must be a string, got {value!r}")
    valid_values = {member.value for member in ResponseTargetType}
    if value not in valid_values:
        raise ValueError(f"target_type must be one of {sorted(valid_values)!r}, got {value!r}")
