"""Orchestrates response-plan optimization from prepared input to persistence."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.response_optimization_result import (
    ResponseOptimizationResult,
    ResponseOptimizationStatus,
)
from src.calculators.response_optimization import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    GeneticResponsePlanOptimizer,
    ResponseOptimizationConfig,
)
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.response_plan import ResponsePlan
from src.repositories.response_plan_repository import ResponsePlanRepository

logger = logging.getLogger(__name__)


class ResponseOptimizationAgent:
    """Coordinates pure GA optimization and append-only response-plan persistence."""

    def __init__(
        self,
        repository: ResponsePlanRepository,
        optimizer: GeneticResponsePlanOptimizer | None = None,
    ) -> None:
        self._repository = repository
        self._optimizer = optimizer or GeneticResponsePlanOptimizer()

    def optimize_from_input(
        self,
        optimization_input: ResponseOptimizationInput,
        *,
        as_of: datetime,
        config: ResponseOptimizationConfig | None = None,
    ) -> ResponseOptimizationResult:
        """Optimize and persist a response plan for an already-normalized input."""
        self._validate_request(optimization_input, as_of)
        effective_config = config or ResponseOptimizationConfig()
        if not isinstance(effective_config, ResponseOptimizationConfig):
            raise ValueError(f"config must be a ResponseOptimizationConfig, got {effective_config!r}")

        try:
            optimization_result = self._optimizer.optimize(optimization_input, effective_config)
            score = optimization_result.score
            plan = ResponsePlan(
                fire_event_id=optimization_input.fire_event_id,
                response_target_set_id=optimization_input.response_target_set_id,
                route_planning_run_id=optimization_input.route_planning_run_id,
                generated_at=as_of,
                status=score.status,
                methodology=METHODOLOGY,
                methodology_version=METHODOLOGY_VERSION,
                random_seed=effective_config.random_seed,
                actions=optimization_result.actions,
                uncovered_target_ids=score.uncovered_target_ids,
                plan_score=score.total_score,
                coverage_score=score.coverage_score,
                average_eta_seconds=score.average_eta_seconds,
            )
            stored = self._repository.save(plan)
            logger.info(
                "Stored optimized response plan %s for FireEvent %s",
                stored.id,
                optimization_input.fire_event_id,
            )
            return ResponseOptimizationResult(
                success=True,
                fire_event_id=optimization_input.fire_event_id,
                response_target_set_id=optimization_input.response_target_set_id,
                route_planning_run_id=optimization_input.route_planning_run_id,
                status=ResponseOptimizationStatus.OPTIMIZED,
                plan_status=plan.status,
                response_plan_id=stored.id,
                action_count=len(plan.actions),
                uncovered_target_count=len(plan.uncovered_target_ids),
                plan_score=plan.plan_score,
                coverage_score=plan.coverage_score,
                average_eta_seconds=plan.average_eta_seconds,
                error_message=None,
            )
        except Exception:
            logger.exception(
                "Response optimization failed for FireEvent %s",
                optimization_input.fire_event_id,
            )
            return ResponseOptimizationResult(
                success=False,
                fire_event_id=optimization_input.fire_event_id,
                response_target_set_id=optimization_input.response_target_set_id,
                route_planning_run_id=optimization_input.route_planning_run_id,
                status=ResponseOptimizationStatus.FAILED,
                plan_status=None,
                response_plan_id=None,
                action_count=0,
                uncovered_target_count=0,
                error_message="Response optimization failed.",
            )

    @staticmethod
    def _validate_request(optimization_input: ResponseOptimizationInput, as_of: datetime) -> None:
        if not isinstance(optimization_input, ResponseOptimizationInput):
            raise ValueError(f"optimization_input must be a ResponseOptimizationInput, got {optimization_input!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
