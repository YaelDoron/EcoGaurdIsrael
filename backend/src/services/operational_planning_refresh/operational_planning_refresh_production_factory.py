"""Production composition for OperationalPlanningRefreshCoordinator.

Builds a real OperationalRefreshOrchestrator (US 4.4, wired the same way
scripts/run_demo_simulation.py already wires one inline - not touched by
this factory) and a real GlobalPlanningRefreshCoordinator (Stage 6, via
global_planning_refresh_production_factory.build_global_planning_refresh_coordinator),
then composes them.

Stage 6 Task 47 (production cutover): this factory used to wire a
ResponsePlanningRefreshOrchestrator (the legacy per-event US 5.4 planner)
as the `planning_refresh` collaborator. It now wires GlobalPlanningRefreshCoordinator
instead, as `global_planning_refresh` - the legacy orchestrator/factory
remain in source (response_planning_production_factory.py) for direct/
historical use, they are simply no longer composed here.

This is the only module in this package that imports Epic 5's routing/GA
package, so it inherits that package's optional osmnx dependency (via
OperationalContextService/RoadNetworkFetcher) - exactly the same dependency
build_global_planning_input_builder itself already has. Importing this
module without osmnx installed will fail; import the coordinator/ports/
result modules directly if only the composition logic is needed (e.g. for
unit tests against fakes).

Mirrors this project's established wiring convention: every repository/
service/agent here already accepts an optional session_factory, so this
module builds the whole graph in one function call. This is a plain
factory function; it exposes no HTTP route. scripts/run_demo_simulation.py
(Stage 6, Task 48) is wired to call THIS coordinator via
SimulationRefreshCoordinator, so the simulation exercises the exact same
production Global GA path.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from src.agents.analysis.fire_severity_assessment_agent import FireSeverityAssessmentAgent
from src.agents.analysis.fire_spread_prediction_agent import FireSpreadPredictionAgent
from src.agents.analysis.response_target_generation_agent import ResponseTargetGenerationAgent
from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.calculators.fire_spread.fire_spread_calculator import FireSpreadCalculator
from src.calculators.response_target.response_target_calculator import ResponseTargetCalculator
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.services.fire_severity.fire_severity_input_service import FireSeverityInputService
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService
from src.services.global_planning.global_planning_refresh_production_factory import (
    build_global_planning_refresh_coordinator,
)
from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
    OperationalPlanningRefreshCoordinator,
)
from src.services.operational_refresh.fire_severity_refresh_orchestrator import FireSeverityRefreshOrchestrator
from src.services.operational_refresh.fire_spread_refresh_orchestrator import FireSpreadRefreshOrchestrator
from src.services.operational_refresh.operational_refresh_orchestrator import OperationalRefreshOrchestrator
from src.services.operational_refresh.resource_status_update_service import ResourceStatusUpdateService
from src.services.response_target.response_target_input_service import ResponseTargetInputService


def build_operational_planning_refresh_coordinator(
    session_factory: sessionmaker[Session] | None = None,
) -> OperationalPlanningRefreshCoordinator:
    """Construct a real, fully-wired OperationalPlanningRefreshCoordinator.

    `session_factory` is forwarded to FireEventRepository (shared by both
    sides) and to build_response_planning_refresh_orchestrator (US 5.4).
    The severity/spread/response-target agents on the US 4.4 side are wired
    the same way scripts/run_demo_simulation.py already wires them - with
    their own repositories' process-wide default session factory - since
    that script does not thread a session_factory through them either;
    this factory does not add that capability, only composes what already
    exists.
    """
    fire_event_repository = FireEventRepository(session_factory)

    severity_input_service = FireSeverityInputService(fire_event_repository=fire_event_repository)
    severity_assessment_repository = FireSeverityAssessmentRepository()
    severity_agent = FireSeverityAssessmentAgent(
        input_service=severity_input_service,
        calculator=FireSeverityCalculator(),
        repository=severity_assessment_repository,
    )
    severity_refresh_orchestrator = FireSeverityRefreshOrchestrator(
        input_service=severity_input_service,
        assessment_agent=severity_agent,
        assessment_repository=severity_assessment_repository,
    )

    spread_input_service = FireSpreadInputService()
    spread_prediction_repository = FireSpreadPredictionRepository()
    spread_agent = FireSpreadPredictionAgent(
        input_service=spread_input_service,
        calculator=FireSpreadCalculator(),
        repository=spread_prediction_repository,
    )
    spread_refresh_orchestrator = FireSpreadRefreshOrchestrator(
        input_service=spread_input_service,
        prediction_agent=spread_agent,
        prediction_repository=spread_prediction_repository,
    )

    response_target_agent = ResponseTargetGenerationAgent(
        input_service=ResponseTargetInputService(),
        calculator=ResponseTargetCalculator(),
        repository=ResponseTargetRepository(),
    )

    resource_repository = FirefightingResourceRepository()
    operational_refresh_orchestrator = OperationalRefreshOrchestrator(
        severity_refresh_orchestrator=severity_refresh_orchestrator,
        spread_refresh_orchestrator=spread_refresh_orchestrator,
        response_target_agent=response_target_agent,
        resource_status_service=ResourceStatusUpdateService(resource_repository),
        resource_repository=resource_repository,
        fire_event_repository=fire_event_repository,
    )

    global_planning_refresh_coordinator = build_global_planning_refresh_coordinator(session_factory=session_factory)

    return OperationalPlanningRefreshCoordinator(
        operational_refresh_orchestrator=operational_refresh_orchestrator,
        global_planning_refresh=global_planning_refresh_coordinator,
    )
