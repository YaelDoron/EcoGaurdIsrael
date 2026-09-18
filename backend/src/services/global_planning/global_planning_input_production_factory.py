"""Production composition for GlobalPlanningInputBuilder (Stage 3 of the
Global Multi-Incident Optimizer refactor).

Mirrors this project's established factory convention (response_planning_
production_factory.py, global_planning_production_factory.py): one
function, `session_factory` forwarded everywhere, no duplicate DB
infrastructure - every repository/service reused here already exists.
"""
from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.global_planning.current_global_assignment_loader import CurrentGlobalAssignmentLoader
from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_candidate_resource_config import GlobalCandidateResourceConfig
from src.services.global_planning.global_incident_demand_builder import GlobalIncidentDemandBuilder
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_route_matrix_builder import GlobalRouteMatrixBuilder
from src.services.operational.operational_context_service import OperationalContextService
from src.services.operational.road_network_fetcher import RoadNetworkFetcher
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver


def build_global_planning_input_builder(
    *,
    candidate_resource_config: GlobalCandidateResourceConfig | None = None,
    severity_demand_policy: SeverityDemandPolicy | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> GlobalPlanningInputBuilder:
    """Construct a real, fully-wired GlobalPlanningInputBuilder.

    `session_factory` is forwarded to every repository, exactly like the
    other Stage 2/1 production factories, so the whole graph can be
    pointed at one database (production or a test session factory).

    `severity_demand_policy` (Stage 5) is forwarded into the
    GlobalIncidentDemandBuilder so callers can point production at a
    specific, versioned demo demand policy without reaching into the
    builder's internals; `None` uses SeverityDemandPolicy's own defaults.
    """
    fire_station_repository = FireStationRepository(session_factory)
    firefighting_resource_repository = FirefightingResourceRepository(session_factory)
    resource_commitment_repository = ResourceCommitmentRepository(session_factory)

    candidate_collector = GlobalCandidateCollector(
        fire_station_repository=fire_station_repository,
        firefighting_resource_repository=firefighting_resource_repository,
        resource_commitment_repository=resource_commitment_repository,
        operational_context_service=OperationalContextService(
            fire_station_repository=fire_station_repository,
            firefighting_resource_repository=firefighting_resource_repository,
        ),
        config=candidate_resource_config,
    )

    incident_demand_builder = GlobalIncidentDemandBuilder(
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory),
        policy=severity_demand_policy,
    )
    current_global_assignment_loader = CurrentGlobalAssignmentLoader(
        current_response_plan_resolver=CurrentResponsePlanResolver(
            ResponsePlanRepository(session_factory), ResponsePlanPlanningStateRepository(session_factory)
        ),
        resource_commitment_repository=resource_commitment_repository,
    )

    return GlobalPlanningInputBuilder(
        global_planning_run_repository=GlobalPlanningRunRepository(session_factory),
        response_target_repository=ResponseTargetRepository(session_factory),
        candidate_collector=candidate_collector,
        incident_demand_builder=incident_demand_builder,
        current_global_assignment_loader=current_global_assignment_loader,
        route_matrix_builder=GlobalRouteMatrixBuilder(),
        road_network_repository=RoadNetworkRepository(),
        road_network_fetcher=RoadNetworkFetcher(),
        session_factory=session_factory,
    )
