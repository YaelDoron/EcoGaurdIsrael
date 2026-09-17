"""Coordinate simulation events through the production US4.4 -> US5.4 bridge."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_detection_result import FireDetectionResult
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.repositories.fire_event_repository import FireEventRepository
from src.services.fire_severity.fire_severity_input_config import SEVERITY_WEATHER_RADIUS_KM
from src.services.operational_planning_refresh.operational_planning_refresh_coordinator import (
    OperationalPlanningRefreshCoordinator,
)
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)
from src.simulation.analysis.simulation_operational_coordinator import SimulationOperationalCoordinator
from src.simulation.analysis.simulation_refresh_result import SimulationRefreshResult
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import SimulationEventExecutionResult
from src.simulation.simulation_scenario import SimulationScenario

logger = logging.getLogger(__name__)

NON_REFRESH_EVENT_REASON = "non_refresh_event"
SOURCE_EVENT_FAILED_REASON = "source_event_failed"
NO_SOURCE_DATA_AVAILABLE_REASON = "no_source_data_available"
NO_DETECTION_RESULT_REASON = "no_detection_result"
DETECTION_FAILED_REASON = "detection_failed"
NO_AFFECTED_FIRE_EVENTS_REASON = "no_affected_fire_events"
NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON = "no_active_fire_events_near_weather"
NO_AVAILABLE_RESOURCE_REASON = "no_available_resource"


class SimulationRefreshCoordinator:
    """Map simulation events to production operational refresh calls."""

    def __init__(
        self,
        *,
        operational_planning_refresh: OperationalPlanningRefreshCoordinator,
        fire_event_repository: FireEventRepository,
        operational_coordinator: SimulationOperationalCoordinator | None = None,
    ) -> None:
        self._operational_planning_refresh = operational_planning_refresh
        self._fire_event_repository = fire_event_repository
        self._operational_coordinator = operational_coordinator
        self._resource_ids_by_selection_key: dict[str, int | str] = {}

    def handle_event(
        self,
        *,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
        detection_result: FireDetectionResult | None = None,
    ) -> SimulationRefreshResult:
        """Run the central production refresh required by one simulation event."""
        self._validate_timestamp(event_timestamp)
        if event.event_type is SimulationEventType.WEATHER:
            return self._handle_weather_event(scenario, event, execution_result, event_timestamp)
        if event.event_type in (SimulationEventType.SATELLITE, SimulationEventType.NEWS):
            return self._handle_detection_event(
                event=event,
                execution_result=execution_result,
                event_timestamp=event_timestamp,
                detection_result=detection_result,
            )
        if event.event_type is SimulationEventType.RESOURCE_STATUS:
            return self._handle_resource_event(scenario, event, execution_result, event_timestamp)
        return SimulationRefreshResult(triggered=False, reason=NON_REFRESH_EVENT_REASON)

    def refresh_fire_events(
        self,
        *,
        fire_event_ids,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> SimulationRefreshResult:
        """Refresh each unique FireEvent once, in deterministic id order, then trigger planning refresh."""
        self._validate_timestamp(as_of)
        normalized_ids = self._normalize_fire_event_ids(fire_event_ids)
        if not normalized_ids:
            return SimulationRefreshResult(triggered=False, reason=NO_AFFECTED_FIRE_EVENTS_REASON)

        operational_results = []
        planning_results = []
        for fire_event_id in normalized_ids:
            try:
                combined_result = self._operational_planning_refresh.refresh_fire_event(
                    fire_event_id=fire_event_id,
                    trigger_type=trigger_type,
                    as_of=as_of,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one incident refresh failure.
                logger.exception("Simulation operational refresh failed for FireEvent %s", fire_event_id)
                operational_results.append(
                    OperationalRefreshResult(
                        trigger_type=trigger_type,
                        status=OperationalRefreshStatus.FAILED,
                        success=False,
                        fire_event_id=fire_event_id,
                        as_of=as_of,
                        error_message=str(exc) or "Operational refresh failed.",
                    )
                )
                continue
            operational_results.append(combined_result.operational_result)
            planning_results.extend(combined_result.planning_results)
        return SimulationRefreshResult(
            triggered=True,
            refresh_results=tuple(operational_results),
            planning_results=tuple(planning_results),
            fire_event_ids=normalized_ids,
        )

    def refresh_resource(
        self,
        *,
        resource_id: int | str,
        event,
        as_of: datetime,
    ) -> SimulationRefreshResult:
        """Refresh one resource, then trigger planning refresh for every active FireEvent."""
        combined_result = self._operational_planning_refresh.refresh_resource(
            resource_id=resource_id,
            new_status=event.resource_status_change.new_status,
            as_of=as_of,
        )
        return SimulationRefreshResult(
            triggered=True,
            refresh_results=(combined_result.operational_result,),
            planning_results=combined_result.planning_results,
        )

    def _handle_weather_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
    ) -> SimulationRefreshResult:
        if not _has_usable_source_data(execution_result):
            reason = SOURCE_EVENT_FAILED_REASON if not execution_result.success else NO_SOURCE_DATA_AVAILABLE_REASON
            return SimulationRefreshResult(triggered=False, reason=reason)

        incident = scenario.get_incident(event.incident_id)
        active_events = self._fire_event_repository.get_active_events_near(
            latitude=incident.location.latitude,
            longitude=incident.location.longitude,
            radius_km=SEVERITY_WEATHER_RADIUS_KM,
            as_of=event_timestamp,
        )
        fire_event_ids = tuple(stored_event.id for stored_event in active_events)
        if not fire_event_ids:
            return SimulationRefreshResult(triggered=False, reason=NO_ACTIVE_FIRE_EVENTS_NEAR_WEATHER_REASON)
        return self.refresh_fire_events(
            fire_event_ids=fire_event_ids,
            trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
            as_of=event_timestamp,
        )

    def _handle_detection_event(
        self,
        *,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
        detection_result: FireDetectionResult | None,
    ) -> SimulationRefreshResult:
        if not _has_usable_source_data(execution_result):
            reason = SOURCE_EVENT_FAILED_REASON if not execution_result.success else NO_SOURCE_DATA_AVAILABLE_REASON
            return SimulationRefreshResult(triggered=False, reason=reason)
        if detection_result is None:
            return SimulationRefreshResult(triggered=False, reason=NO_DETECTION_RESULT_REASON)
        if not detection_result.success:
            return SimulationRefreshResult(triggered=False, reason=DETECTION_FAILED_REASON)
        if not detection_result.event_ids:
            return SimulationRefreshResult(triggered=False, reason=NO_AFFECTED_FIRE_EVENTS_REASON)
        return self.refresh_fire_events(
            fire_event_ids=detection_result.event_ids,
            trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
            as_of=event_timestamp,
        )

    def _handle_resource_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        execution_result: SimulationEventExecutionResult,
        event_timestamp: datetime,
    ) -> SimulationRefreshResult:
        if not execution_result.success:
            return SimulationRefreshResult(triggered=False, reason=SOURCE_EVENT_FAILED_REASON)
        resource_id = self._resolve_resource_id(scenario, event)
        if resource_id is None:
            return SimulationRefreshResult(triggered=False, reason=NO_AVAILABLE_RESOURCE_REASON)
        return self.refresh_resource(resource_id=resource_id, event=event, as_of=event_timestamp)

    def _resolve_resource_id(self, scenario: SimulationScenario, event: SimulationEvent) -> int | str | None:
        resource_change = event.resource_status_change
        if resource_change.resource_id is not None:
            return resource_change.resource_id

        selection_key = resource_change.selection_key or f"{event.incident_id}:resource"
        if selection_key in self._resource_ids_by_selection_key:
            return self._resource_ids_by_selection_key[selection_key]

        if self._operational_coordinator is None:
            return None
        incident = scenario.get_incident(event.incident_id)
        selected = self._operational_coordinator.select_available_resource(
            incident.location.latitude,
            incident.location.longitude,
        )
        if selected is None:
            return None
        self._resource_ids_by_selection_key[selection_key] = selected.id
        return selected.id

    @staticmethod
    def _normalize_fire_event_ids(fire_event_ids) -> tuple[int, ...]:
        normalized_ids = tuple(sorted(set(fire_event_ids)))
        for fire_event_id in normalized_ids:
            if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
                raise ValueError(f"fire_event_ids must contain positive integer ids, got {fire_event_id!r}.")
        return normalized_ids

    @staticmethod
    def _validate_timestamp(as_of: datetime) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}.")


def _has_usable_source_data(execution_result: SimulationEventExecutionResult) -> bool:
    return execution_result.saved_count > 0 or execution_result.duplicates_skipped > 0
