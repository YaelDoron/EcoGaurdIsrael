"""Central production orchestration for operational state refreshes."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_severity_assessment_agent import FireSeverityAssessmentAgent
from src.agents.analysis.response_target_generation_agent import ResponseTargetGenerationAgent
from src.agents.analysis.response_target_generation_result import ResponseTargetGenerationStatus
from src.models.fire_event_status import FireEventStatus
from src.models.firefighting_resource import FirefightingResource
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.resource_status import ResourceStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.services.operational_refresh.fire_spread_refresh_orchestrator import FireSpreadRefreshOrchestrator
from src.services.operational_refresh.fire_spread_refresh_result import (
    FireSpreadRefreshHorizonStatus,
    FireSpreadRefreshResult,
)
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)
from src.services.operational_refresh.resource_status_update_result import ResourceStatusUpdateStatus
from src.services.operational_refresh.resource_status_update_service import ResourceStatusUpdateService

logger = logging.getLogger(__name__)

_ENVIRONMENTAL_TRIGGER_TYPES = frozenset(
    {
        OperationalRefreshTriggerType.WEATHER_UPDATE,
        OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        OperationalRefreshTriggerType.SEVERITY_UPDATE,
    }
)
_ACTIVE_FIRE_EVENT_STATUSES = frozenset({FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED})


class OperationalRefreshOrchestrator:
    """Route operational changes to the existing production refresh components."""

    def __init__(
        self,
        *,
        severity_agent: FireSeverityAssessmentAgent,
        spread_refresh_orchestrator: FireSpreadRefreshOrchestrator,
        response_target_agent: ResponseTargetGenerationAgent,
        resource_status_service: ResourceStatusUpdateService,
        resource_repository: FirefightingResourceRepository,
        fire_event_repository: FireEventRepository,
    ) -> None:
        self._severity_agent = severity_agent
        self._spread_refresh_orchestrator = spread_refresh_orchestrator
        self._response_target_agent = response_target_agent
        self._resource_status_service = resource_status_service
        self._resource_repository = resource_repository
        self._fire_event_repository = fire_event_repository

    def refresh_fire_event(
        self,
        *,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> OperationalRefreshResult:
        """Refresh current severity/spread/targets for one FireEvent operational change."""
        self._validate_fire_event_request(fire_event_id, trigger_type, as_of)

        severity_result = None
        try:
            if self._should_run_severity(fire_event_id, trigger_type):
                severity_result = self._severity_agent.assess(fire_event_id, as_of)
        except Exception as exc:  # noqa: BLE001 - conservative orchestration failure boundary.
            logger.exception("Severity refresh failed for FireEvent %s", fire_event_id)
            return OperationalRefreshResult(
                trigger_type=trigger_type,
                status=OperationalRefreshStatus.FAILED,
                success=False,
                fire_event_id=fire_event_id,
                as_of=as_of,
                error_message=str(exc) or "Severity refresh failed.",
            )

        spread_result = self._spread_refresh_orchestrator.refresh(
            fire_event_id=fire_event_id,
            trigger_type=trigger_type,
            as_of=as_of,
        )
        if spread_result.failed:
            return OperationalRefreshResult(
                trigger_type=trigger_type,
                status=OperationalRefreshStatus.FAILED,
                success=False,
                fire_event_id=fire_event_id,
                as_of=as_of,
                severity_result=severity_result,
                spread_refresh_result=spread_result,
                error_message="Fire-spread refresh failed.",
            )

        target_result = self._response_target_agent.generate(fire_event_id=fire_event_id, as_of=as_of)
        if target_result.status is ResponseTargetGenerationStatus.FAILED:
            return OperationalRefreshResult(
                trigger_type=trigger_type,
                status=OperationalRefreshStatus.FAILED,
                success=False,
                fire_event_id=fire_event_id,
                as_of=as_of,
                severity_result=severity_result,
                spread_refresh_result=spread_result,
                response_target_result=target_result,
                error_message=target_result.error_message or "Response-target refresh failed.",
            )

        return OperationalRefreshResult(
            trigger_type=trigger_type,
            status=self._environmental_status(spread_result, target_result.status),
            success=True,
            fire_event_id=fire_event_id,
            as_of=as_of,
            severity_result=severity_result,
            spread_refresh_result=spread_result,
            response_target_result=target_result,
        )

    def refresh_resource(
        self,
        *,
        resource_id: int | str,
        new_status: ResourceStatus,
    ) -> OperationalRefreshResult:
        """Update one resource status and return the affected station's available resources."""
        self._validate_resource_request(resource_id, new_status)

        result = self._resource_status_service.update_status(
            resource_id=resource_id,
            new_status=new_status,
        )
        available_resources = ()
        if result.resource is not None:
            available_resources = tuple(
                self._to_domain_resource(resource)
                for resource in self._resource_repository.get_available_resources([str(result.resource.station_id)])
            )

        if result.status is ResourceStatusUpdateStatus.UPDATED:
            status = OperationalRefreshStatus.RESOURCE_UPDATED
            success = True
            error_message = None
        elif result.status is ResourceStatusUpdateStatus.NO_OP:
            status = OperationalRefreshStatus.RESOURCE_NO_OP
            success = True
            error_message = None
        elif result.status is ResourceStatusUpdateStatus.NOT_FOUND:
            status = OperationalRefreshStatus.RESOURCE_NOT_FOUND
            success = False
            error_message = f"Firefighting resource {resource_id!r} was not found."
        else:
            status = OperationalRefreshStatus.FAILED
            success = False
            error_message = result.error_message or "Resource status update failed."

        return OperationalRefreshResult(
            trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
            status=status,
            success=success,
            resource_status_result=result,
            available_resources=available_resources,
            error_message=error_message,
        )

    def _should_run_severity(
        self,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
    ) -> bool:
        if trigger_type is OperationalRefreshTriggerType.SEVERITY_UPDATE:
            return False
        if trigger_type is OperationalRefreshTriggerType.WEATHER_UPDATE:
            return True
        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        return (
            stored_event is not None
            and stored_event.event.status in _ACTIVE_FIRE_EVENT_STATUSES
        )

    @staticmethod
    def _environmental_status(
        spread_result: FireSpreadRefreshResult,
        target_status: ResponseTargetGenerationStatus,
    ) -> OperationalRefreshStatus:
        if target_status is ResponseTargetGenerationStatus.INACTIVE_EVENT:
            return OperationalRefreshStatus.INACTIVE_EVENT
        if spread_result.horizon_results and all(
            result.status is FireSpreadRefreshHorizonStatus.NO_OP
            for result in spread_result.horizon_results
        ):
            return OperationalRefreshStatus.NO_OP
        return OperationalRefreshStatus.REFRESHED

    @staticmethod
    def _to_domain_resource(resource) -> FirefightingResource:
        if isinstance(resource, FirefightingResource):
            return resource
        return FirefightingResource(id=resource.id, station_id=resource.station_id, status=resource.status)

    @staticmethod
    def _validate_fire_event_request(
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}.")
        if trigger_type not in _ENVIRONMENTAL_TRIGGER_TYPES:
            raise ValueError(f"refresh_fire_event does not accept trigger_type {trigger_type!r}.")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}.")

    @staticmethod
    def _validate_resource_request(resource_id: int | str, new_status: ResourceStatus) -> None:
        if (
            not isinstance(resource_id, (int, str))
            or isinstance(resource_id, bool)
            or (isinstance(resource_id, int) and resource_id <= 0)
            or (isinstance(resource_id, str) and not resource_id.strip())
        ):
            raise ValueError(f"resource_id must be a positive int or non-empty string, got {resource_id!r}.")
        if not isinstance(new_status, ResourceStatus):
            raise ValueError(f"new_status must be a ResourceStatus, got {new_status!r}.")
