"""Coordinate active wildfire detection after simulated direct-evidence persistence."""
from __future__ import annotations

from datetime import datetime

from src.agents.analysis.fire_detection_agent import FireDetectionAgent
from src.simulation.analysis.simulation_fire_detection_result import SimulationFireDetectionResult
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import SimulationEventExecutionResult
from src.simulation.simulation_scenario import SimulationScenario

NON_DETECTION_EVENT_REASON = "non_detection_event"
SOURCE_EVENT_FAILED_REASON = "source_event_failed"
NO_SOURCE_DATA_AVAILABLE_REASON = "no_source_data_available"

_DETECTION_EVENT_TYPES = {SimulationEventType.SATELLITE, SimulationEventType.NEWS}


class SimulationFireDetectionCoordinator:
    """Trigger active wildfire detection after satellite/news simulation events."""

    def __init__(self, detection_agent: FireDetectionAgent) -> None:
        self._detection_agent = detection_agent

    def handle_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
    ) -> SimulationFireDetectionResult:
        """Run detection after persisted satellite/news source data is available.

        Duplicate source rows are considered usable because the same persisted
        evidence remains queryable by FireDetectionEvidenceService.
        """
        if event.event_type not in _DETECTION_EVENT_TYPES:
            return SimulationFireDetectionResult(
                triggered=False,
                detection_result=None,
                reason=NON_DETECTION_EVENT_REASON,
            )

        if execution_result.saved_count <= 0 and execution_result.duplicates_skipped <= 0:
            reason = SOURCE_EVENT_FAILED_REASON if not execution_result.success else NO_SOURCE_DATA_AVAILABLE_REASON
            return SimulationFireDetectionResult(triggered=False, detection_result=None, reason=reason)

        detection_result = self._detection_agent.detect(as_of=event_timestamp)
        return SimulationFireDetectionResult(
            triggered=True,
            detection_result=detection_result,
            reason=None,
        )
