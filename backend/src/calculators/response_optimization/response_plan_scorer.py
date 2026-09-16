"""Pure response-plan scoring for US 5.2."""
from __future__ import annotations

from src.calculators.response_optimization.response_optimization_config import (
    ETA_REFERENCE_SECONDS,
    ResponseOptimizationConfig,
)
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.plan_score_breakdown import PlanScoreBreakdown, derive_response_plan_status
from src.models.response_action import ResponseAction
from src.models.response_optimization_input import ResponseOptimizationInput


class ResponsePlanScoringError(ValueError):
    """Raised when a candidate action set cannot be scored."""


class ResponsePlanScorer:
    """Evaluate one-to-one resource-to-target response-action candidates."""

    def __init__(self, config: ResponseOptimizationConfig | None = None) -> None:
        self._config = config or ResponseOptimizationConfig()

    def eta_factor(self, travel_time_seconds: float) -> float:
        """Return V1 ETA utility factor.

        `ETA_REFERENCE_SECONDS` is an engineering normalization constant, not
        an official wildfire response-time standard.
        """
        if (
            isinstance(travel_time_seconds, bool)
            or not isinstance(travel_time_seconds, (int, float))
            or travel_time_seconds < 0
        ):
            raise ResponsePlanScoringError(
                f"travel_time_seconds must be a finite non-negative number, got {travel_time_seconds!r}"
            )
        if travel_time_seconds == float("inf") or travel_time_seconds != travel_time_seconds:
            raise ResponsePlanScoringError(
                f"travel_time_seconds must be a finite non-negative number, got {travel_time_seconds!r}"
            )
        return 1.0 / (1.0 + float(travel_time_seconds) / self._config.eta_reference_seconds)

    def evaluate(
        self,
        optimization_input: ResponseOptimizationInput,
        actions: tuple[ResponseAction, ...],
    ) -> PlanScoreBreakdown:
        """Score candidate actions against normalized optimization input."""
        if not isinstance(optimization_input, ResponseOptimizationInput):
            raise ResponsePlanScoringError(
                f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}"
            )
        normalized_actions = self._normalize_actions(actions)
        target_by_id = {target.response_target_id: target for target in optimization_input.targets}
        route_by_id = {route.route_result_id: route for route in optimization_input.route_options}
        resource_ids = {resource.resource_id for resource in optimization_input.resources}
        action_routes = self._validate_actions(
            actions=normalized_actions,
            target_by_id=target_by_id,
            route_by_id=route_by_id,
            resource_ids=resource_ids,
        )

        raw_fitness = 0.0
        covered_priority = 0.0
        eta_values = []
        covered_target_ids = set()
        for action, route in action_routes:
            target = target_by_id[action.response_target_id]
            eta = route.travel_time_seconds
            if eta is None:
                raise ResponsePlanScoringError("reachable route option is missing travel_time_seconds.")
            raw_fitness += target.priority_score * self.eta_factor(eta)
            covered_priority += target.priority_score
            eta_values.append(float(eta))
            covered_target_ids.add(action.response_target_id)

        total_priority = sum(target.priority_score for target in optimization_input.targets)
        if total_priority > 0:
            total_score = 100.0 * raw_fitness / total_priority
            coverage_score = 100.0 * covered_priority / total_priority
        else:
            total_score = 0.0
            coverage_score = 0.0
            raw_fitness = 0.0
            covered_priority = 0.0

        average_eta_seconds = None
        if eta_values:
            average_eta_seconds = sum(eta_values) / len(eta_values)

        uncovered_target_ids = tuple(
            target.response_target_id
            for target in optimization_input.targets
            if target.response_target_id not in covered_target_ids
        )
        status = derive_response_plan_status(
            total_target_count=len(optimization_input.targets),
            covered_target_count=len(covered_target_ids),
        )

        return PlanScoreBreakdown(
            total_score=total_score,
            raw_fitness=raw_fitness,
            coverage_score=coverage_score,
            average_eta_seconds=average_eta_seconds,
            total_priority=total_priority,
            covered_priority=covered_priority,
            covered_target_count=len(covered_target_ids),
            total_target_count=len(optimization_input.targets),
            uncovered_target_ids=uncovered_target_ids,
            status=status,
        )

    @staticmethod
    def _normalize_actions(actions) -> tuple[ResponseAction, ...]:
        try:
            action_tuple = tuple(actions)
        except TypeError as exc:
            raise ResponsePlanScoringError("actions must be iterable.") from exc
        for action in action_tuple:
            if not isinstance(action, ResponseAction):
                raise ResponsePlanScoringError(f"actions must contain ResponseAction items, got {action!r}")
        return tuple(sorted(action_tuple, key=lambda action: action.ordering_key))

    @staticmethod
    def _validate_actions(
        *,
        actions: tuple[ResponseAction, ...],
        target_by_id: dict[int, OptimizationTarget],
        route_by_id: dict[int, OptimizationRouteOption],
        resource_ids: set[int | str],
    ) -> tuple[tuple[ResponseAction, OptimizationRouteOption], ...]:
        resource_assignments = set()
        target_assignments = set()
        resource_target_pairs = set()
        action_routes: list[tuple[ResponseAction, OptimizationRouteOption]] = []

        for action in actions:
            if action.resource_id in resource_assignments:
                raise ResponsePlanScoringError(f"resource {action.resource_id!r} is assigned more than once.")
            if action.response_target_id in target_assignments:
                raise ResponsePlanScoringError(
                    f"response target {action.response_target_id!r} is assigned more than once."
                )
            pair = (action.resource_id, action.response_target_id)
            if pair in resource_target_pairs:
                raise ResponsePlanScoringError(f"duplicate resource-target assignment {pair!r}.")
            if action.resource_id not in resource_ids:
                raise ResponsePlanScoringError(f"action references unknown resource_id {action.resource_id!r}.")
            if action.response_target_id not in target_by_id:
                raise ResponsePlanScoringError(
                    f"action references unknown response_target_id {action.response_target_id!r}."
                )
            if action.route_result_id not in route_by_id:
                raise ResponsePlanScoringError(f"action references unknown route_result_id {action.route_result_id!r}.")

            route = route_by_id[action.route_result_id]
            if route.resource_id != action.resource_id:
                raise ResponsePlanScoringError(
                    "route_result_id does not belong to the action resource, got "
                    f"{route.resource_id!r} for {action.resource_id!r}."
                )
            if route.response_target_id != action.response_target_id:
                raise ResponsePlanScoringError(
                    "route_result_id does not belong to the action target, got "
                    f"{route.response_target_id!r} for {action.response_target_id!r}."
                )
            if not route.is_reachable:
                raise ResponsePlanScoringError(f"route_result_id {route.route_result_id!r} is unreachable.")
            if route.travel_time_seconds is None:
                raise ResponsePlanScoringError(f"route_result_id {route.route_result_id!r} has no ETA.")

            resource_assignments.add(action.resource_id)
            target_assignments.add(action.response_target_id)
            resource_target_pairs.add(pair)
            action_routes.append((action, route))

        return tuple(action_routes)


def eta_factor(travel_time_seconds: float, eta_reference_seconds: float = ETA_REFERENCE_SECONDS) -> float:
    """Convenience pure ETA factor function using the V1 formula."""
    return ResponsePlanScorer(
        ResponseOptimizationConfig(eta_reference_seconds=eta_reference_seconds)
    ).eta_factor(travel_time_seconds)
