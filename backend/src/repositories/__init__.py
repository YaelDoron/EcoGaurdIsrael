"""Persistence layer bridging Task 2 domain models and SQLAlchemy ORM models."""

from src.repositories.exceptions import (
    FireDangerAssessmentRepositoryError,
    FireEventRepositoryError,
    GlobalPlanningRunRepositoryError,
    ResourceCommitmentRepositoryError,
    ResponseTargetRepositoryError,
    ResponsePlanRepositoryError,
    RoutePlanningRepositoryError,
    FireSeverityAssessmentRepositoryError,
    NewsRepositoryError,
    SatelliteHotspotRepositoryError,
    WeatherRepositoryError,
    WeatherStationNotStoredError,
)
from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.repositories.global_planning_run_repository import (
    GlobalPlanningRunRepository,
    StoredGlobalPlanningRun,
    StoredGlobalPlanningRunEvent,
)
from src.repositories.news_repository import NewsRepository, SaveNewsReportResult, StoredWildfireReport
from src.repositories.plan_comparison_repository import PlanComparisonRepository, StoredPlanComparison
from src.repositories.road_network_repository import RoadNetworkRepository
from src.repositories.response_target_repository import (
    ResponseTargetRepository,
    StoredResponseTarget,
    StoredResponseTargetSet,
)
from src.repositories.response_plan_repository import ResponsePlanRepository, StoredResponsePlan
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.route_planning_repository import RoutePlanningRepository, StoredRoutePlanningRun
from src.repositories.satellite_hotspot_repository import (
    SaveHotspotResult,
    SatelliteHotspotRepository,
    StoredSatelliteHotspot,
)
from src.repositories.weather_repository import (
    SaveObservationResult,
    StoredWeatherObservation,
    WeatherRepository,
)

__all__ = [
    "WeatherRepository",
    "SaveObservationResult",
    "StoredWeatherObservation",
    "SatelliteHotspotRepository",
    "SaveHotspotResult",
    "StoredSatelliteHotspot",
    "NewsRepository",
    "SaveNewsReportResult",
    "StoredWildfireReport",
    "FireDangerAssessmentRepository",
    "StoredFireDangerAssessment",
    "FireEventRepository",
    "StoredFireEvent",
    "FireStationRepository",
    "FirefightingResourceRepository",
    "FireSeverityAssessmentRepository",
    "StoredFireSeverityAssessment",
    "RoadNetworkRepository",
    "ResponseTargetRepository",
    "StoredResponseTarget",
    "StoredResponseTargetSet",
    "PlanComparisonRepository",
    "StoredPlanComparison",
    "ResponsePlanRepository",
    "StoredResponsePlan",
    "ResourceCommitmentRepository",
    "ResourceCommitmentRepositoryError",
    "GlobalPlanningRunRepository",
    "StoredGlobalPlanningRun",
    "StoredGlobalPlanningRunEvent",
    "GlobalPlanningRunRepositoryError",
    "RoutePlanningRepository",
    "StoredRoutePlanningRun",
    "WeatherRepositoryError",
    "WeatherStationNotStoredError",
    "SatelliteHotspotRepositoryError",
    "NewsRepositoryError",
    "FireDangerAssessmentRepositoryError",
    "FireEventRepositoryError",
    "FireSeverityAssessmentRepositoryError",
    "ResponseTargetRepositoryError",
    "ResponsePlanRepositoryError",
    "RoutePlanningRepositoryError",
]
