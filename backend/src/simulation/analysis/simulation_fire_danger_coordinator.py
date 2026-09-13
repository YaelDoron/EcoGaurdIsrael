"""Coordinate fire-danger assessment after simulated weather persistence."""
from __future__ import annotations

from datetime import datetime
import re

from src.agents.analysis.fire_danger_assessment_agent import FireDangerAssessmentAgent
from src.models.assessment_area import AssessmentArea
from src.simulation.analysis.simulation_fire_danger_result import SimulationFireDangerResult
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import SimulationEventExecutionResult
from src.simulation.simulation_locations import get_simulation_location_key
from src.simulation.simulation_scenario import SimulationScenario

ASSESSMENT_RADIUS_KM = 5.0

NON_WEATHER_EVENT_REASON = "non_weather_event"
WEATHER_EVENT_FAILED_REASON = "weather_event_failed"
NO_WEATHER_SAVED_REASON = "no_weather_saved"


class SimulationFireDangerCoordinator:
    """Trigger fire-danger analysis for completed simulated weather events."""

    def __init__(self, assessment_agent: FireDangerAssessmentAgent) -> None:
        self._assessment_agent = assessment_agent

    def handle_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
    ) -> SimulationFireDangerResult:
        """Run fire-danger assessment when a simulated event makes weather data available."""
        if event.event_type is not SimulationEventType.WEATHER:
            return SimulationFireDangerResult(
                triggered=False,
                assessment_result=None,
                reason=NON_WEATHER_EVENT_REASON,
            )

        if execution_result.saved_count <= 0 and execution_result.duplicates_skipped <= 0:
            reason = WEATHER_EVENT_FAILED_REASON if not execution_result.success else NO_WEATHER_SAVED_REASON
            return SimulationFireDangerResult(triggered=False, assessment_result=None, reason=reason)

        incident = scenario.get_incident(event.incident_id)
        assessment_area = _assessment_area_from_location(incident.location)
        assessment_result = self._assessment_agent.assess(area=assessment_area, as_of=event_timestamp)
        return SimulationFireDangerResult(
            triggered=True,
            assessment_result=assessment_result,
            reason=None,
        )


def _assessment_area_from_location(location) -> AssessmentArea:  # noqa: ANN001
    location_key = get_simulation_location_key(location)
    area_id = f"simulation-{location_key}" if location_key is not None else _custom_area_id(location)
    return AssessmentArea(
        id=area_id,
        name=location.name,
        latitude=location.latitude,
        longitude=location.longitude,
        radius_km=ASSESSMENT_RADIUS_KM,
    )


def _custom_area_id(location) -> str:  # noqa: ANN001
    slug = re.sub(r"[^a-z0-9]+", "-", location.name.strip().lower()).strip("-")
    return slug or f"simulation-area-{location.latitude:.6f}-{location.longitude:.6f}"
