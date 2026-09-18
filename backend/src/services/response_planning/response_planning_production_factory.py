"""Production composition for the real ResponsePlanningRefreshOrchestrator (Epic 5, US 5.4, Task 6).

Mirrors this project's established wiring convention: every repository,
service, and agent involved already accepts an optional session_factory /
collaborator with a real default, so this module constructs the full graph
with one function call rather than requiring callers to hand-assemble each
piece - matching how backend/scripts/run_demo_simulation.py already wires
OperationalRefreshOrchestrator inline (there is no dependency-injection
framework or composition container elsewhere in this project to follow
instead). This is a plain factory function; it exposes no HTTP route.

`baseline_plan_scorer` defaults to the real production scoring path
(ResponsePlanBaselineScorerAdapter wrapping the shared ResponsePlanScorer) as
of Task 6.1 - the US 5.3 data-shape gap that previously made this impossible
without fabricating a scorer has been closed (see baseline_plan_calculator.py
and baseline_plan_evaluator.py's Task 6.1 notes). The parameter remains as an
optional override purely as a test/customization seam; production callers
never need to supply one.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from src.agents.analysis.response_optimization_agent import ResponseOptimizationAgent
from src.agents.routing.route_planning_agent import RoutePlanningAgent
from src.calculators.baseline_plan.baseline_plan_evaluator import BaselinePlanScorer
from src.calculators.response_optimization.response_optimization_config import DEFAULT_RANDOM_SEED
from src.calculators.response_optimization.response_plan_scorer import ResponsePlanScorer
from src.calculators.routing.dijkstra_calculator import DijkstraCalculator
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.baseline_comparison.baseline_comparison_production_readers import (
    ResponsePlanOptimizedPlanReaderAdapter,
    RoutePlanningRunReaderAdapter,
)
from src.services.baseline_comparison.baseline_comparison_service import BaselineComparisonService
from src.services.baseline_comparison.response_plan_baseline_scorer_adapter import (
    ResponsePlanBaselineScorerAdapter,
)
from src.services.operational.operational_context_service import OperationalContextService
from src.services.response_planning.baseline_comparison_collaborator_adapter import (
    BaselineComparisonCollaboratorAdapter,
)
from src.services.response_planning.planning_effective_state_builder import PlanningEffectiveStateBuilder
from src.services.response_planning.response_optimization_collaborator_adapter import (
    ResponseOptimizationCollaboratorAdapter,
)
from src.services.response_planning.response_planning_refresh_orchestrator import (
    ResponsePlanningRefreshOrchestrator,
)
from src.services.resource_reservation import CrossEventReservedResourceResolver, ResponsePlanActivationService
from src.services.routing.node_mapping_service import NodeMappingService


def build_response_planning_refresh_orchestrator(
    *,
    baseline_plan_scorer: BaselinePlanScorer | None = None,
    session_factory: sessionmaker[Session] | None = None,
    optimization_seed: int = DEFAULT_RANDOM_SEED,
) -> ResponsePlanningRefreshOrchestrator:
    """Construct a real, fully-wired ResponsePlanningRefreshOrchestrator.

    `session_factory` is forwarded to every repository so callers can point
    the whole graph at one database (e.g. a test SQLite engine); omitting it
    lets each repository fall back to its own process-wide default.

    `baseline_plan_scorer` defaults to the real
    `ResponsePlanBaselineScorerAdapter(ResponsePlanScorer())` production
    path; pass an override only for tests/customization.
    """
    if baseline_plan_scorer is None:
        baseline_plan_scorer = ResponsePlanBaselineScorerAdapter(ResponsePlanScorer())

    response_target_repository = ResponseTargetRepository(session_factory)
    route_planning_repository = RoutePlanningRepository(session_factory)
    response_plan_repository = ResponsePlanRepository(session_factory)
    plan_comparison_repository = PlanComparisonRepository(session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(session_factory)
    operational_context_service = OperationalContextService(
        fire_station_repository=FireStationRepository(session_factory),
        firefighting_resource_repository=FirefightingResourceRepository(session_factory),
        cross_event_reserved_resource_resolver=CrossEventReservedResourceResolver(
            fire_event_repository=FireEventRepository(session_factory),
            response_plan_repository=response_plan_repository,
            resource_commitment_repository=resource_commitment_repository,
        ),
    )

    routing_agent = RoutePlanningAgent(
        response_target_repository=response_target_repository,
        operational_context_service=operational_context_service,
        node_mapping_service=NodeMappingService(),
        dijkstra_calculator=DijkstraCalculator(),
        route_planning_repository=route_planning_repository,
        session_factory=session_factory,
    )

    optimization_adapter = ResponseOptimizationCollaboratorAdapter(
        optimization_agent=ResponseOptimizationAgent(repository=response_plan_repository),
        route_planning_repository=route_planning_repository,
        response_target_repository=response_target_repository,
    )

    baseline_service = BaselineComparisonService(
        optimized_plan_reader=ResponsePlanOptimizedPlanReaderAdapter(response_plan_repository),
        route_planning_run_reader=RoutePlanningRunReaderAdapter(route_planning_repository),
        scorer=baseline_plan_scorer,
        response_target_set_reader=response_target_repository,
        plan_comparison_repository=plan_comparison_repository,
    )
    baseline_adapter = BaselineComparisonCollaboratorAdapter(
        baseline_comparison_service=baseline_service,
        plan_comparison_repository=plan_comparison_repository,
    )

    activation_service = ResponsePlanActivationService(
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=resource_commitment_repository,
        session_factory=session_factory,
    )

    return ResponsePlanningRefreshOrchestrator(
        planning_state_builder=PlanningEffectiveStateBuilder(
            response_target_repository=response_target_repository,
            operational_context_service=operational_context_service,
        ),
        routing_collaborator=routing_agent,
        optimization_collaborator=optimization_adapter,
        baseline_collaborator=baseline_adapter,
        activation_collaborator=activation_service,
        fire_event_repository=FireEventRepository(session_factory),
        response_plan_repository=response_plan_repository,
        response_plan_planning_state_repository=ResponsePlanPlanningStateRepository(session_factory),
        resource_commitment_repository=resource_commitment_repository,
        plan_comparison_repository=plan_comparison_repository,
        optimization_seed=optimization_seed,
    )
