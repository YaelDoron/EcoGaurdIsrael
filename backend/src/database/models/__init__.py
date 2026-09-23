"""SQLAlchemy ORM models (persistence layer). See src.models for the Task 2 domain dataclasses."""

from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_severity_assessment_satellite_input_db import (
    FireSeverityAssessmentSatelliteInputDB,
)
from src.database.models.fire_severity_assessment_weather_input_db import (
    FireSeverityAssessmentWeatherInputDB,
)
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.fire_spread_prediction_weather_input_db import (
    FireSpreadPredictionWeatherInputDB,
)
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB
from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.database.models.plan_comparison_db import PlanComparisonDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_action_db import ResponseActionDB
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.database.models.wildfire_report_db import WildfireReportDB

__all__ = [
    "WeatherStationDB",
    "WeatherObservationDB",
    "SatelliteHotspotDB",
    "WildfireReportDB",
    "FireDangerAssessmentDB",
    "FireDangerAssessmentWeatherInputDB",
    "FireEventDB",
    "FireEventMLAssessmentDB",
    "FireEventSatelliteEvidenceDB",
    "FireEventNewsEvidenceDB",
    "FireStationDB",
    "FirefightingResourceDB",
    "FireSeverityAssessmentDB",
    "FireSeverityAssessmentWeatherInputDB",
    "FireSeverityAssessmentSatelliteInputDB",
    "FireSpreadPredictionDB",
    "FireSpreadPredictionCellDB",
    "FireSpreadPredictionWeatherInputDB",
    "GlobalPlanningRunDB",
    "GlobalPlanningRunEventDB",
    "GraphNodeDB",
    "GraphEdgeDB",
    "PlanComparisonDB",
    "ResponseTargetSetDB",
    "ResponseTargetDB",
    "RoutePlanningRunDB",
    "RouteResultDB",
    "ResponsePlanDB",
    "ResponseActionDB",
    "ResponsePlanPlanningStateDB",
    "ResponsePlanUncoveredTargetDB",
    "ResourceCommitmentDB",
]
