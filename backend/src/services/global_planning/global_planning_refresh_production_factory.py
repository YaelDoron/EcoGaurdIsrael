"""Production composition for GlobalPlanningRefreshCoordinator (Stage 6 of
the Global Multi-Incident Optimizer refactor).

Mirrors this project's established factory convention (response_planning_
production_factory.py, global_planning_input_production_factory.py): one
function, `session_factory` forwarded everywhere, no duplicate DB
infrastructure - every repository/service reused here already exists.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from src.calculators.global_response_optimization.global_assignment_stability_policy import (
    GlobalAssignmentStabilityPolicy,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.global_planning.global_planning_input_production_factory import (
    build_global_planning_input_builder,
)
from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshCoordinator
from src.services.global_planning.global_response_plan_activation_service import GlobalResponsePlanActivationService


def build_global_planning_refresh_coordinator(
    *,
    config: GlobalResponseOptimizationConfig | None = None,
    demand_scoring_policy: GlobalDemandScoringPolicy | None = None,
    severity_demand_policy: SeverityDemandPolicy | None = None,
    stability_policy: GlobalAssignmentStabilityPolicy | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> GlobalPlanningRefreshCoordinator:
    """Construct a real, fully-wired GlobalPlanningRefreshCoordinator - the
    authoritative production entry point for the Global GA (Stage 6, Task
    18). `session_factory` is forwarded to every repository, exactly like
    the other Stage 1-5 production factories, so the whole graph can be
    pointed at one database (production or a test session factory).
    """
    global_planning_run_repository = GlobalPlanningRunRepository(session_factory)
    activation_service = GlobalResponsePlanActivationService(
        response_plan_repository=ResponsePlanRepository(session_factory),
        response_plan_planning_state_repository=ResponsePlanPlanningStateRepository(session_factory),
        route_planning_repository=RoutePlanningRepository(session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(session_factory),
        global_planning_run_repository=global_planning_run_repository,
        session_factory=session_factory,
    )

    return GlobalPlanningRefreshCoordinator(
        fire_event_repository=FireEventRepository(session_factory),
        global_planning_run_repository=global_planning_run_repository,
        input_builder=build_global_planning_input_builder(session_factory=session_factory),
        optimization_service=GlobalResponseOptimizationService(),
        activation_service=activation_service,
        config=config,
        demand_scoring_policy=demand_scoring_policy,
        severity_demand_policy=severity_demand_policy,
        stability_policy=stability_policy,
    )
