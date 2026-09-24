"""Coordinate fire-severity assessment after meaningful simulated input changes."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.agents.analysis.fire_severity_assessment_agent import FireSeverityAssessmentAgent
from src.models.fire_event_response_eligibility import is_response_eligible
from src.repositories.fire_event_repository import FireEventRepository
from src.services.fire_severity.fire_severity_input_config import SEVERITY_WEATHER_RADIUS_KM
from src.simulation.analysis.simulation_fire_severity_result import SimulationFireSeverityResult
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import SimulationEventExecutionResult
from src.simulation.simulation_scenario import SimulationScenario

logger = logging.getLogger(__name__)

NON_SEVERITY_EVENT_REASON = "non_severity_event"
SOURCE_EVENT_FAILED_REASON = "source_event_failed"
NO_SOURCE_DATA_AVAILABLE_REASON = "no_source_data_available"
NO_DETECTION_RESULT_REASON = "no_detection_result"
DETECTION_FAILED_REASON = "detection_failed"
NO_AFFECTED_FIRE_EVENTS_REASON = "no_affected_fire_events"
NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON = "no_active_fire_events_near_weather"
NO_RESPONSE_ELIGIBLE_FIRE_EVENTS_REASON = "no_response_eligible_fire_events"


class SimulationFireSeverityCoordinator:
    """Trigger active-fire severity assessments from simulation events."""

    def __init__(
        self,
        severity_agent: FireSeverityAssessmentAgent,
        fire_event_repository: FireEventRepository,
    ) -> None:
        self._severity_agent = severity_agent
        self._fire_event_repository = fire_event_repository

    def handle_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
        detection_result: FireDetectionResult | None = None,
    ) -> SimulationFireSeverityResult:
        """Run severity after satellite or weather events change Severity v1 inputs."""
        if event.event_type is SimulationEventType.WEATHER:
            return self._handle_weather_event(scenario, event, execution_result, event_timestamp)
        if event.event_type is SimulationEventType.SATELLITE:
            return self._handle_satellite_event(execution_result, event_timestamp, detection_result)
        return SimulationFireSeverityResult(triggered=False, reason=NON_SEVERITY_EVENT_REASON)

    def _handle_weather_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
    ) -> SimulationFireSeverityResult:
        if not _has_usable_source_data(execution_result):
            reason = SOURCE_EVENT_FAILED_REASON if not execution_result.success else NO_SOURCE_DATA_AVAILABLE_REASON
            return SimulationFireSeverityResult(triggered=False, reason=reason)

        incident = scenario.get_incident(event.incident_id)
        # Task 9A: severity is response-pipeline work: only response-eligible (CONFIRMED) fires are assessed.
        active_events = self._fire_event_repository.get_response_eligible_events_near(
            latitude=incident.location.latitude,
            longitude=incident.location.longitude,
            radius_km=SEVERITY_WEATHER_RADIUS_KM,
            as_of=event_timestamp,
        )
        fire_event_ids = tuple(stored_event.id for stored_event in active_events)
        if not fire_event_ids:
            return SimulationFireSeverityResult(
                triggered=False,
                reason=NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON,
            )
        return self._assess_fire_events(fire_event_ids, event_timestamp)

    def _handle_satellite_event(
        self,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
        detection_result: FireDetectionResult | None,
    ) -> SimulationFireSeverityResult:
        if not _has_usable_source_data(execution_result):
            reason = SOURCE_EVENT_FAILED_REASON if not execution_result.success else NO_SOURCE_DATA_AVAILABLE_REASON
            return SimulationFireSeverityResult(triggered=False, reason=reason)
        if detection_result is None:
            return SimulationFireSeverityResult(triggered=False, reason=NO_DETECTION_RESULT_REASON)
        if not detection_result.success:
            return SimulationFireSeverityResult(triggered=False, reason=DETECTION_FAILED_REASON)

        fire_event_ids = tuple(sorted(set(detection_result.event_ids)))
        if not fire_event_ids:
            return SimulationFireSeverityResult(triggered=False, reason=NO_AFFECTED_FIRE_EVENTS_REASON)
        # Task 9A: SUSPECTED events (active for monitoring only) are not assessed; only response-eligible ones are.
        eligible_ids = self._response_eligible_ids(fire_event_ids)
        if not eligible_ids:
            return SimulationFireSeverityResult(triggered=False, reason=NO_RESPONSE_ELIGIBLE_FIRE_EVENTS_REASON)
        return self._assess_fire_events(eligible_ids, event_timestamp)

    def _response_eligible_ids(self, fire_event_ids: tuple[int, ...]) -> tuple[int, ...]:
        eligible = []
        for fire_event_id in fire_event_ids:
            stored_event = self._fire_event_repository.get_by_id(fire_event_id)
            if stored_event is not None and is_response_eligible(stored_event.event.status):
                eligible.append(fire_event_id)
        return tuple(eligible)

    def _assess_fire_events(
        self,
        fire_event_ids: tuple[int, ...],
        event_timestamp: datetime,
    ) -> SimulationFireSeverityResult:
        assessment_results = []
        failed_fire_event_ids = []
        error_messages = []
        for fire_event_id in tuple(sorted(set(fire_event_ids))):
            try:
                assessment_results.append(
                    self._severity_agent.assess(
                        fire_event_id=fire_event_id,
                        assessed_at=event_timestamp,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - record per-event analysis failure after source persistence.
                logger.exception("Simulation fire-severity assessment failed for FireEvent %s", fire_event_id)
                failed_fire_event_ids.append(fire_event_id)
                error_messages.append(str(exc) or "Fire-severity assessment failed.")

        return SimulationFireSeverityResult(
            triggered=True,
            assessment_results=tuple(assessment_results),
            failed_fire_event_ids=tuple(failed_fire_event_ids),
            error_messages=tuple(error_messages),
        )


def _has_usable_source_data(execution_result: SimulationEventExecutionResult) -> bool:
    return execution_result.saved_count > 0 or execution_result.duplicates_skipped > 0
