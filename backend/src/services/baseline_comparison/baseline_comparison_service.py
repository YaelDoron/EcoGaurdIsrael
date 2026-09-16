"""Orchestration for US 5.3 Task 5: compare one optimized plan against its baseline.

`BaselineComparisonService.compare(response_plan_id=...)` is the frozen
Task 0 entry point. It introduces NO new algorithm -- it is the conductor
that wires together already-complete Company 3 components in order:

1. load the EXACT persisted optimized plan by `response_plan_id`, by
   default through `ResponsePlanRepositoryOptimizedPlanReader`, a real
   adapter over US 5.2's `ResponsePlanRepository` (`OptimizedPlanReader`
   is the structural port -- see `baseline_comparison_ports.py`);
2. load the EXACT `RoutePlanningRun` and `ResponseTargetSet` it references,
   by default through `RoutePlanningRepositoryRunReader` (a real adapter
   over US 5.1's `RoutePlanningRepository`) and the real US 4.3
   `ResponseTargetRepository` directly;
3. validate the planning chain is self-consistent (same FireEvent, same
   routing run id, same target set id);
4. convert the persisted snapshot into Task 1's existing pure input types
   (`TargetOrder`, `RouteCandidate`) using ONLY already-persisted values --
   no ETA/reachability/priority/routing recalculation;
5. call Task 2's `BaselinePlanEvaluator` exactly once (it already runs
   Task 1's greedy allocation and, by default via
   `ResponsePlanScorerBaselineAdapter`, the real US 5.2
   `ResponsePlanScorer` reached through `ResponseOptimizationInputService`);
6. build `OptimizedPlanEvaluation` from the optimized plan's
   already-computed, already-persisted score (never rescored here) and
   call Task 3's `BaselinePlanComparisonCalculator` exactly once;
7. call Task 4's `PlanComparisonRepository.save(...)` exactly once;
8. return the `PlanComparison` Task 3 produced.

This module does not reimplement any Task 1-4 logic, does not invoke any
detection/severity/spread/routing/GA agent, and does not query "current"
resource/target/routing state -- the baseline must use the exact same
planning snapshot the optimized plan used. The real adapters live in
`baseline_comparison_adapters.py`; the `Protocol` ports in
`baseline_comparison_ports.py` keep the service decoupled from any one
persistence implementation and remain overridable (e.g. by tests).
"""
from __future__ import annotations

import logging

from src.calculators.baseline_plan.baseline_plan_calculator import RouteCandidate, TargetOrder
from src.calculators.baseline_plan.baseline_plan_comparison_calculator import (
    BaselinePlanComparisonCalculator,
    OptimizedPlanEvaluation,
    PlanComparison,
)
from src.calculators.baseline_plan.baseline_plan_evaluator import BaselinePlanEvaluator, BaselinePlanScorer
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_target_repository import ResponseTargetRepository, StoredResponseTargetSet
from src.services.baseline_comparison.baseline_comparison_adapters import (
    ResponsePlanRepositoryOptimizedPlanReader,
    ResponsePlanScorerBaselineAdapter,
    RoutePlanningRepositoryRunReader,
)
from src.services.baseline_comparison.baseline_comparison_ports import (
    OptimizedPlanLike,
    OptimizedPlanReader,
    RoutePlanningRunLike,
    RoutePlanningRunReader,
    ResponseTargetSetReader,
)

logger = logging.getLogger(__name__)


class BaselineComparisonServiceError(Exception):
    """Raised for Task 5 orchestration failures: missing records or planning-chain mismatches."""


class BaselineComparisonService:
    """Conductor that produces and persists one `PlanComparison` for one optimized plan."""

    def __init__(
        self,
        *,
        optimized_plan_reader: OptimizedPlanReader | None = None,
        route_planning_run_reader: RoutePlanningRunReader | None = None,
        scorer: BaselinePlanScorer | None = None,
        response_target_set_reader: ResponseTargetSetReader | None = None,
        plan_comparison_repository: PlanComparisonRepository | None = None,
        baseline_evaluator: BaselinePlanEvaluator | None = None,
        comparison_calculator: BaselinePlanComparisonCalculator | None = None,
    ) -> None:
        self._optimized_plan_reader = optimized_plan_reader or ResponsePlanRepositoryOptimizedPlanReader()
        self._route_planning_run_reader = route_planning_run_reader or RoutePlanningRepositoryRunReader()
        self._scorer = scorer or ResponsePlanScorerBaselineAdapter()
        self._response_target_set_reader = (
            response_target_set_reader if response_target_set_reader is not None else ResponseTargetRepository()
        )
        self._plan_comparison_repository = (
            plan_comparison_repository if plan_comparison_repository is not None else PlanComparisonRepository()
        )
        self._baseline_evaluator = baseline_evaluator if baseline_evaluator is not None else BaselinePlanEvaluator()
        self._comparison_calculator = (
            comparison_calculator if comparison_calculator is not None else BaselinePlanComparisonCalculator()
        )

    def compare(self, *, response_plan_id: int) -> PlanComparison:
        """Build, evaluate, compare, and persist the baseline for one exact optimized plan.

        Raises `BaselineComparisonServiceError` if the plan/run/target-set
        cannot be found or the planning chain is inconsistent -- in every
        such case nothing is evaluated and no `PlanComparison` is saved.
        """
        self._validate_response_plan_id(response_plan_id)

        optimized_plan = self._optimized_plan_reader.get_by_id(response_plan_id)
        if optimized_plan is None:
            raise BaselineComparisonServiceError(f"ResponsePlan {response_plan_id!r} was not found.")

        routing_run = self._route_planning_run_reader.get_by_id(optimized_plan.route_planning_run_id)
        if routing_run is None:
            raise BaselineComparisonServiceError(
                f"RoutePlanningRun {optimized_plan.route_planning_run_id!r} was not found."
            )

        stored_target_set = self._response_target_set_reader.get_by_id(optimized_plan.response_target_set_id)
        if stored_target_set is None:
            raise BaselineComparisonServiceError(
                f"ResponseTargetSet {optimized_plan.response_target_set_id!r} was not found."
            )

        self._validate_planning_chain(optimized_plan, routing_run, stored_target_set)

        targets = self._build_target_order(stored_target_set)
        route_candidates = self._build_route_candidates(routing_run, stored_target_set)

        baseline_result = self._baseline_evaluator.evaluate(
            fire_event_id=optimized_plan.fire_event_id,
            route_planning_run_id=optimized_plan.route_planning_run_id,
            response_target_set_id=optimized_plan.response_target_set_id,
            targets=targets,
            route_candidates=route_candidates,
            scorer=self._scorer,
        )

        optimized_evaluation = OptimizedPlanEvaluation(
            optimized_plan_id=optimized_plan.id,
            fire_event_id=optimized_plan.fire_event_id,
            route_planning_run_id=optimized_plan.route_planning_run_id,
            response_target_set_id=optimized_plan.response_target_set_id,
            score=optimized_plan.score,
        )

        comparison = self._comparison_calculator.compare(optimized=optimized_evaluation, baseline=baseline_result)

        stored_comparison = self._plan_comparison_repository.save(comparison)
        logger.info(
            "Stored PlanComparison %s for ResponsePlan %s (FireEvent %s)",
            stored_comparison.id,
            response_plan_id,
            optimized_plan.fire_event_id,
        )

        return comparison

    @staticmethod
    def _validate_planning_chain(
        optimized_plan: OptimizedPlanLike,
        routing_run: RoutePlanningRunLike,
        stored_target_set: StoredResponseTargetSet,
    ) -> None:
        if routing_run.id != optimized_plan.route_planning_run_id:
            raise BaselineComparisonServiceError(
                "Loaded RoutePlanningRun id does not match the optimized plan's "
                f"route_planning_run_id: loaded={routing_run.id!r} "
                f"expected={optimized_plan.route_planning_run_id!r}."
            )
        if stored_target_set.id != optimized_plan.response_target_set_id:
            raise BaselineComparisonServiceError(
                "Loaded ResponseTargetSet id does not match the optimized plan's "
                f"response_target_set_id: loaded={stored_target_set.id!r} "
                f"expected={optimized_plan.response_target_set_id!r}."
            )
        if routing_run.fire_event_id != optimized_plan.fire_event_id:
            raise BaselineComparisonServiceError(
                "RoutePlanningRun fire_event_id does not match the optimized plan's "
                f"fire_event_id: run={routing_run.fire_event_id!r} "
                f"plan={optimized_plan.fire_event_id!r}."
            )
        if stored_target_set.target_set.fire_event_id != optimized_plan.fire_event_id:
            raise BaselineComparisonServiceError(
                "ResponseTargetSet fire_event_id does not match the optimized plan's "
                f"fire_event_id: target_set={stored_target_set.target_set.fire_event_id!r} "
                f"plan={optimized_plan.fire_event_id!r}."
            )
        if routing_run.response_target_set_id != optimized_plan.response_target_set_id:
            raise BaselineComparisonServiceError(
                "RoutePlanningRun response_target_set_id does not match the optimized plan's "
                f"response_target_set_id: run={routing_run.response_target_set_id!r} "
                f"plan={optimized_plan.response_target_set_id!r}."
            )

    @staticmethod
    def _build_target_order(stored_target_set: StoredResponseTargetSet) -> tuple[TargetOrder, ...]:
        # Persisted response_target_id / target_order / target_type /
        # priority_score ONLY, copied unchanged from the exact stored
        # target -- Task 1's own ordering/allocation logic is unaffected;
        # target_type/priority_score are carried through solely for the
        # shared scorer (Task 6.1).
        return tuple(
            TargetOrder(
                response_target_id=stored_target.id,
                target_order=stored_target.target_order,
                target_type=stored_target.target.target_type,
                priority_score=stored_target.target.priority_score,
            )
            for stored_target in stored_target_set.targets
        )

    @staticmethod
    def _build_route_candidates(
        routing_run: RoutePlanningRunLike,
        stored_target_set: StoredResponseTargetSet,
    ) -> tuple[RouteCandidate, ...]:
        # Persisted route_result_id / resource_id / response_target_id /
        # status / travel_time_seconds / distance_meters ONLY, copied
        # unchanged from the exact stored route -- no ETA/path/distance/node
        # mapping is calculated here.
        valid_target_ids = {stored_target.id for stored_target in stored_target_set.targets}
        valid_resource_ids = set(routing_run.resource_ids)

        candidates = []
        for route_result in routing_run.route_results:
            if route_result.response_target_id not in valid_target_ids:
                raise BaselineComparisonServiceError(
                    f"RouteResult {route_result.id!r} references response_target_id "
                    f"{route_result.response_target_id!r}, which is not part of ResponseTargetSet "
                    f"{stored_target_set.id!r}."
                )
            if route_result.resource_id not in valid_resource_ids:
                raise BaselineComparisonServiceError(
                    f"RouteResult {route_result.id!r} references resource_id "
                    f"{route_result.resource_id!r}, which is not part of RoutePlanningRun "
                    f"{routing_run.id!r}'s resource snapshot."
                )
            candidates.append(
                RouteCandidate(
                    route_result_id=route_result.id,
                    resource_id=route_result.resource_id,
                    response_target_id=route_result.response_target_id,
                    status=route_result.status,
                    travel_time_seconds=route_result.travel_time_seconds,
                    distance_meters=route_result.distance_meters,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _validate_response_plan_id(response_plan_id: int) -> None:
        if isinstance(response_plan_id, bool) or not isinstance(response_plan_id, int) or response_plan_id <= 0:
            raise ValueError(f"response_plan_id must be a positive integer, got {response_plan_id!r}.")
