"""EcoGuard internal domain models (plain dataclasses, no persistence)."""

from src.models.active_fire_events import (
    ActiveFireEventSeveritySummary,
    ActiveFireEventSummary,
    ActiveFireEventsResult,
)
from src.models.assessment_area import AssessmentArea
from src.models.fire_danger_areas import (
    FireDangerAreaAssessmentSummary,
    FireDangerAreaSnapshot,
    FireDangerAreasResult,
)
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_detail import (
    FireDangerAssessmentDetail,
    FireDangerAssessmentWeatherInputSummary,
)
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_calculation import FireDangerCalculation
from src.models.fire_danger_input import FireDangerInput
from src.models.fire_danger_input_result import FireDangerInputResult
from src.models.fire_danger_input_status import FireDangerInputStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_decision import FireDetectionDecision
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_detection_status import FireDetectionStatus
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.news_text_analysis import NewsTextAnalysis
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.models.fire_station import FireStation
from src.models.firefighting_resource import FirefightingResource
from src.models.resource_status import ResourceStatus
from src.models.fire_severity_calculation import FireSeverityCalculation
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_input import FireSeverityInput
from src.models.fire_severity_input_result import FireSeverityInputResult
from src.models.fire_severity_input_status import FireSeverityInputStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.fire_spread_calculation import FireSpreadCalculation
from src.models.fire_spread_effective_state import FireSpreadEffectiveState
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason
from src.models.fire_spread_input_result import FireSpreadInputResult
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_prediction import FireSpreadPrediction, FireSpreadPredictionCell
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.models.demand_source import DemandSource
from src.models.global_allocation_slot import GlobalAllocationSlot
from src.models.global_event_optimization_result import GlobalEventOptimizationResult
from src.models.global_event_resource_demand_result import GlobalEventResourceDemandResult
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_optimization_result import GlobalOptimizationResult
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_resource_shortage import GlobalResourceShortage
from src.models.global_response_action import GlobalResponseAction
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event import GlobalPlanningRunEvent
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.operations_overview import (
    FireDangerActivityPreview,
    FireEventActivityPreview,
    FireSeverityActivityPreview,
    GlobalPlanningRunActivityPreview,
    NewsReportActivityPreview,
    OperationsActivityFeed,
    OperationsActivityFeedItem,
    OperationsOverviewSnapshot,
    OperationsSimulationSummary,
    SatelliteHotspotActivityPreview,
)
from src.models.operations_activity import (
    FireDangerActivityDetail,
    FireEventActivityDetail,
    FireEventActivityDetails,
    FireEventEvidenceRefs,
    FireEventSeverityReference,
    FireSeverityActivityDetail,
    FireSeverityActivityDetails,
    GlobalPlanningRunActivityDetail,
    GlobalPlanningRunActivityDetails,
    GlobalPlanningRunMemberSummary,
    NewsReportActivityDetail,
    OperationsActivityDetail,
    OperationsActivityLocation,
    OperationsActivityType,
    SatelliteHotspotActivityDetail,
)
from src.models.optimization_resource import OptimizationResource
from src.models.optimization_route_option import OptimizationRouteOption
from src.models.optimization_target import OptimizationTarget
from src.models.plan_score_breakdown import PlanScoreBreakdown, derive_response_plan_status
from src.models.planning_effective_state import (
    PlanningEffectiveState,
    PlanningResourceState,
    PlanningTargetState,
)
from src.models.planning_effective_state_result import PlanningEffectiveStateResult
from src.models.planning_effective_state_status import PlanningEffectiveStateStatus
from src.models.predicted_risk_target_candidate import PredictedRiskTargetCandidate
from src.models.response_action import ResponseAction
from src.models.response_optimization_input import ResponseOptimizationInput
from src.models.response_plan_chromosome import ResponsePlanChromosome
from src.models.response_plan import ResponsePlan
from src.models.response_plan_details import (
    BaselineComparisonDetails,
    OptimizationConfigDetails,
    ResponseActionDetails,
    ResponsePlanDetails,
)
from src.models.response_plan_status import ResponsePlanStatus
from src.models.response_target import ResponseTarget
from src.models.response_target_input import ResponseTargetInput
from src.models.response_target_input_result import ResponseTargetInputResult
from src.models.response_target_input_status import ResponseTargetInputStatus
from src.models.resource_commitment import ResourceCommitment
from src.models.response_target_set import ResponseTargetSet
from src.models.response_target_type import ResponseTargetType
from src.models.routing import (
    RoutePlanningRun,
    RouteResult,
    RouteStatus,
    RoutingResource,
    RoutingTarget,
    StoredRouteResult,
)
from src.models.satellite_hotspot import SatelliteHotspot
from src.models.vegetation_data import VegetationData
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

__all__ = [
    "ActiveFireEventSeveritySummary",
    "ActiveFireEventSummary",
    "ActiveFireEventsResult",
    "WeatherStation",
    "WeatherObservation",
    "SatelliteHotspot",
    "WildfireReport",
    "NewsTextAnalysis",
    "NewsWildfireSignalStrength",
    "FireDangerInput",
    "FireDangerAssessment",
    "FireDangerAssessmentStatus",
    "FireDangerCalculation",
    "FireDangerLevel",
    "FireDangerAreaAssessmentSummary",
    "FireDangerAreaSnapshot",
    "FireDangerAreasResult",
    "FireDangerAssessmentDetail",
    "FireDangerAssessmentWeatherInputSummary",
    "AssessmentArea",
    "FireDangerInputResult",
    "FireDangerInputStatus",
    "FireEvidenceType",
    "FireDetectionEvidence",
    "FireEvidenceRef",
    "FireDetectionCandidate",
    "FireDetectionStatus",
    "FireDetectionDecision",
    "FireEvent",
    "FireEventStatus",
    "FireStation",
    "FirefightingResource",
    "ResourceStatus",
    "FireSeverityInput",
    "FireSeverityAssessment",
    "FireSeverityAssessmentStatus",
    "FireSeverityInputResult",
    "FireSeverityInputStatus",
    "FireSeverityLevel",
    "FireSeverityCalculation",
    "FireSpreadCalculation",
    "FireSpreadEffectiveState",
    "FireSpreadFuelClass",
    "FireSpreadInput",
    "FireSpreadInsufficientDataReason",
    "FireSpreadInputResult",
    "FireSpreadInputStatus",
    "FireSpreadPrediction",
    "FireSpreadPredictionCell",
    "FireSpreadPredictionStatus",
    "PredictedRiskTargetCandidate",
    "ResponseTarget",
    "ResponseTargetInput",
    "ResponseTargetInputResult",
    "ResponseTargetInputStatus",
    "ResponseTargetSet",
    "ResponseTargetType",
    "VegetationData",
    "GlobalPlanningRun",
    "GlobalPlanningRunEvent",
    "GlobalPlanningRunEventStatus",
    "GlobalPlanningRunStatus",
    "GlobalPlanningInput",
    "GlobalPlanningResource",
    "GlobalPlanningTarget",
    "GlobalRouteMatrix",
    "GlobalRouteOption",
    "GlobalAllocationSlot",
    "GlobalResponsePlanChromosome",
    "GlobalResponseAction",
    "GlobalEventOptimizationResult",
    "GlobalOptimizationResult",
    "DemandSource",
    "GlobalIncidentDemand",
    "GlobalEventResourceDemandResult",
    "GlobalResourceShortage",
    "GraphNode",
    "GraphEdge",
    "OperationalRefreshTriggerType",
    "OperationsActivityType",
    "OperationsActivityLocation",
    "OperationsActivityDetail",
    "FireDangerActivityDetail",
    "SatelliteHotspotActivityDetail",
    "NewsReportActivityDetail",
    "FireEventEvidenceRefs",
    "FireEventSeverityReference",
    "FireEventActivityDetails",
    "FireEventActivityDetail",
    "FireSeverityActivityDetails",
    "FireSeverityActivityDetail",
    "GlobalPlanningRunMemberSummary",
    "GlobalPlanningRunActivityDetails",
    "GlobalPlanningRunActivityDetail",
    "OperationsSimulationSummary",
    "FireDangerActivityPreview",
    "SatelliteHotspotActivityPreview",
    "NewsReportActivityPreview",
    "FireEventActivityPreview",
    "FireSeverityActivityPreview",
    "GlobalPlanningRunActivityPreview",
    "OperationsActivityFeedItem",
    "OperationsActivityFeed",
    "OperationsOverviewSnapshot",
    "RouteStatus",
    "RoutingResource",
    "RoutingTarget",
    "RouteResult",
    "RoutePlanningRun",
    "StoredRouteResult",
    "OptimizationTarget",
    "OptimizationResource",
    "OptimizationRouteOption",
    "ResponseOptimizationInput",
    "ResponseAction",
    "ResponsePlanChromosome",
    "ResponsePlanStatus",
    "ResourceCommitment",
    "ResponsePlan",
    "ResponseActionDetails",
    "BaselineComparisonDetails",
    "OptimizationConfigDetails",
    "ResponsePlanDetails",
    "PlanScoreBreakdown",
    "derive_response_plan_status",
    "PlanningTargetState",
    "PlanningResourceState",
    "PlanningEffectiveState",
    "PlanningEffectiveStateStatus",
    "PlanningEffectiveStateResult",
]
