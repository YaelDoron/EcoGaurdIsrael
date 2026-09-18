"""Production composition for GlobalPlanningOrchestrator (Stage 2 of the
Global Multi-Incident Optimizer refactor).

Mirrors response_planning_production_factory.py's convention: one function,
`session_factory` forwarded everywhere so callers can point the whole graph
at one database. Reuses build_response_planning_refresh_orchestrator
unchanged for the child orchestrator (Task 10: no copied/duplicated
per-event logic).

Task 18: NOT wired into any existing production trigger. Callers (a
script, a test, a future explicit endpoint) construct this directly; no
existing planning-refresh call site is changed by this module's existence.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from src.calculators.baseline_plan.baseline_plan_evaluator import BaselinePlanScorer
from src.calculators.response_optimization.response_optimization_config import DEFAULT_RANDOM_SEED
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.services.global_planning.global_planning_orchestrator import GlobalPlanningOrchestrator
from src.services.operational.operational_context_service import OperationalContextService
from src.services.response_planning.planning_effective_state_builder import PlanningEffectiveStateBuilder
from src.services.response_planning.response_planning_production_factory import (
    build_response_planning_refresh_orchestrator,
)
from src.services.resource_reservation import CrossEventReservedResourceResolver


def build_global_planning_orchestrator(
    *,
    baseline_plan_scorer: BaselinePlanScorer | None = None,
    session_factory: sessionmaker[Session] | None = None,
    optimization_seed: int = DEFAULT_RANDOM_SEED,
) -> GlobalPlanningOrchestrator:
    """Construct a real, fully-wired GlobalPlanningOrchestrator.

    `session_factory` is forwarded to every repository, exactly like
    build_response_planning_refresh_orchestrator - both graphs end up
    pointed at the same database when called with the same session_factory
    (as production and tests both do).
    """
    child_orchestrator = build_response_planning_refresh_orchestrator(
        baseline_plan_scorer=baseline_plan_scorer,
        session_factory=session_factory,
        optimization_seed=optimization_seed,
    )

    fire_event_repository = FireEventRepository(session_factory)
    response_plan_repository = ResponsePlanRepository(session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(session_factory)

    # A second PlanningEffectiveStateBuilder instance, wired the same way as
    # the one inside child_orchestrator - the builder is a pure reader with
    # no internal state, so this is behaviorally identical to sharing one
    # instance (Task 6: reuse its semantics, not necessarily its object).
    operational_context_service = OperationalContextService(
        fire_station_repository=FireStationRepository(session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(session_factory),
        cross_event_reserved_resource_resolver=CrossEventReservedResourceResolver(
            fire_event_repository=fire_event_repository,
            response_plan_repository=response_plan_repository,
            resource_commitment_repository=resource_commitment_repository,
        ),
    )
    planning_state_builder = PlanningEffectiveStateBuilder(
        response_target_repository=ResponseTargetRepository(session_factory),
        operational_context_service=operational_context_service,
    )

    return GlobalPlanningOrchestrator(
        child_orchestrator=child_orchestrator,
        fire_event_repository=fire_event_repository,
        resource_commitment_repository=resource_commitment_repository,
        response_plan_repository=response_plan_repository,
        planning_state_builder=planning_state_builder,
        global_planning_run_repository=GlobalPlanningRunRepository(session_factory),
    )
