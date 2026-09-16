"""Pure response-plan domain model for US 5.2."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.models.optimization_validation import (
    validate_non_empty_string,
    validate_optional_finite_non_negative_number,
    validate_positive_int,
)
from src.models.response_action import ResponseAction
from src.models.response_plan_status import ResponsePlanStatus


@dataclass(frozen=True)
class ResponsePlan:
    """Generated response-allocation recommendation.

    This pure model has no persistence, routing, or scoring behavior. Score
    values are externally supplied by later optimization tasks.
    """

    fire_event_id: int
    response_target_set_id: int
    route_planning_run_id: int
    generated_at: datetime
    status: ResponsePlanStatus
    methodology: str
    methodology_version: str
    random_seed: int
    actions: tuple[ResponseAction, ...]
    uncovered_target_ids: tuple[int, ...]
    plan_score: float | None = None
    coverage_score: float | None = None
    average_eta_seconds: float | None = None

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        validate_positive_int("response_target_set_id", self.response_target_set_id)
        validate_positive_int("route_planning_run_id", self.route_planning_run_id)
        if not isinstance(self.generated_at, datetime) or self.generated_at.tzinfo is None:
            raise ValueError(f"generated_at must be a timezone-aware datetime, got {self.generated_at!r}")
        if not isinstance(self.status, ResponsePlanStatus):
            raise ValueError(f"status must be a ResponsePlanStatus, got {self.status!r}")
        validate_non_empty_string("methodology", self.methodology)
        validate_non_empty_string("methodology_version", self.methodology_version)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError(f"random_seed must be an integer, got {self.random_seed!r}")

        actions = self._normalize_actions(self.actions)
        uncovered_target_ids = self._normalize_uncovered_target_ids(self.uncovered_target_ids)
        self._validate_status_invariants(self.status, actions, uncovered_target_ids)
        self._validate_coverage_invariants(actions, uncovered_target_ids)

        validate_optional_finite_non_negative_number("plan_score", self.plan_score)
        validate_optional_finite_non_negative_number("coverage_score", self.coverage_score)
        validate_optional_finite_non_negative_number("average_eta_seconds", self.average_eta_seconds)

        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "uncovered_target_ids", uncovered_target_ids)

    @staticmethod
    def _normalize_actions(actions) -> tuple[ResponseAction, ...]:
        try:
            action_tuple = tuple(actions)
        except TypeError as exc:
            raise ValueError("actions must be iterable.") from exc
        for action in action_tuple:
            if not isinstance(action, ResponseAction):
                raise ValueError(f"actions must contain ResponseAction items, got {action!r}")

        resource_ids = [action.resource_id for action in action_tuple]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("a resource may appear in at most one response action.")
        action_keys = [(action.resource_id, action.response_target_id) for action in action_tuple]
        if len(action_keys) != len(set(action_keys)):
            raise ValueError("actions must not duplicate resource-target assignments.")
        return tuple(sorted(action_tuple, key=lambda action: action.ordering_key))

    @staticmethod
    def _normalize_uncovered_target_ids(uncovered_target_ids) -> tuple[int, ...]:
        try:
            ids = tuple(uncovered_target_ids)
        except TypeError as exc:
            raise ValueError("uncovered_target_ids must be iterable.") from exc
        for target_id in ids:
            validate_positive_int("uncovered_target_id", target_id)
        if len(ids) != len(set(ids)):
            raise ValueError("uncovered_target_ids must be unique.")
        return tuple(sorted(ids))

    @staticmethod
    def _validate_status_invariants(
        status: ResponsePlanStatus,
        actions: tuple[ResponseAction, ...],
        uncovered_target_ids: tuple[int, ...],
    ) -> None:
        if status is ResponsePlanStatus.COMPLETE and uncovered_target_ids:
            raise ValueError("COMPLETE response plans must not include uncovered_target_ids.")
        if status is ResponsePlanStatus.PARTIAL:
            if not actions:
                raise ValueError("PARTIAL response plans require at least one action.")
            if not uncovered_target_ids:
                raise ValueError("PARTIAL response plans require at least one uncovered target.")
        if status is ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS and actions:
            raise ValueError("NO_FEASIBLE_ASSIGNMENTS response plans must not include actions.")

    @staticmethod
    def _validate_coverage_invariants(
        actions: tuple[ResponseAction, ...],
        uncovered_target_ids: tuple[int, ...],
    ) -> None:
        covered_target_ids = {action.response_target_id for action in actions}
        uncovered_ids = set(uncovered_target_ids)
        overlap = covered_target_ids & uncovered_ids
        if overlap:
            raise ValueError(f"covered and uncovered target sets must not overlap, got {sorted(overlap)!r}.")
