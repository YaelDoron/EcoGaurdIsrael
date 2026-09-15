"""EcoGuard internal domain models (plain dataclasses, no persistence)."""

from src.models.assessment_area import AssessmentArea
from src.models.fire_danger_assessment import FireDangerAssessment
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
from src.models.fire_station import FireStation
from src.models.firefighting_resource import FirefightingResource
from src.models.resource_status import ResourceStatus
from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

__all__ = [
    "WeatherStation",
    "WeatherObservation",
    "SatelliteHotspot",
    "WildfireReport",
    "FireDangerInput",
    "FireDangerAssessment",
    "FireDangerAssessmentStatus",
    "FireDangerCalculation",
    "FireDangerLevel",
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
]
