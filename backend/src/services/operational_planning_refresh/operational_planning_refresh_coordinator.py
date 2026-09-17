"""US 4.4 -> US 5.4 orchestration bridge.

OperationalPlanningRefreshCoordinator is a thin composition layer: it calls
OperationalRefreshOrchestrator (US 4.4), and - only if that refresh
succeeded - calls a PlanningRefreshPort (ResponsePlanningRefreshOrchestrator
in production, US 5.4) for every FireEvent whose planning may depend on
what changed. It contains no severity/spread/target logic of its own, and
no Dijkstra/GA/baseline logic of its own: whether a planning cycle actually
needs to rerun is decided entirely by the injected planning collaborator's
own effective-state fingerprint, never re-derived or duplicated here.
"""
from __future__ import annotations

from datetime import datetime
import logging

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.resource_status import ResourceStatus
from src.repositories.fire_event_repository import FireEventRepository
from src.services.operational_planning_refresh.operational_planning_refresh_ports import PlanningRefreshPort
from src.services.operational_planning_refresh.operational_planning_refresh_result import (
    OperationalPlanningRefreshResult,
)
from src.services.operational_refresh.operational_refresh_orchestrator import OperationalRefreshOrchestrator
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)

logger = logging.getLogger(__name__)


class OperationalPlanningRefreshCoordinator:
    """Trigger a US 5.4 planning refresh after a US 4.4 operational refresh."""

    def __init__(
        self,
        *,
        operational_refresh_orchestrator: OperationalRefreshOrchestrator,
        planning_refresh: PlanningRefreshPort,
        fire_event_repository: FireEventRepository | None = None,
    ) -> None:
        self._operational_refresh_orchestrator = operational_refresh_orchestrator
        self._planning_refresh = planning_refresh
        self._fire_event_repository = fire_event_repository or FireEventRepository()

    def refresh_fire_event(
        self,
        *,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> OperationalPlanningRefreshResult:
        """Run one US 4.4 environmental/FireEvent refresh, then a planning refresh for the same FireEvent.

        Planning refresh only runs after severity/spread/response-target
        refresh has completed (severity, then spread, then response
        targets, in that existing order), so it observes the newest
        persisted planning inputs. It never runs after an operational
        failure. Whether the planning refresh actually reruns routing/
        optimization/baseline, or is a NO_OP, is entirely
        ResponsePlanningRefreshOrchestrator's own decision via its planning
        fingerprint - not decided or precomputed here.
        """
        operational_result = self._run_operational_fire_event_refresh(fire_event_id, trigger_type, as_of)
        if not operational_result.success:
            return OperationalPlanningRefreshResult(operational_result=operational_result, planning_results=())

        planning_result = self._planning_refresh.refresh(fire_event_id=fire_event_id, as_of=as_of)
        return OperationalPlanningRefreshResult(
            operational_result=operational_result,
            planning_results=(planning_result,),
        )

    def refresh_resource(
        self,
        *,
        resource_id: int | str,
        new_status: ResourceStatus,
        as_of: datetime,
    ) -> OperationalPlanningRefreshResult:
        """Run one US 4.4 resource-status update, then a planning refresh for every active FireEvent.

        Runs no Fire Detection, Fire Severity, Fire Spread, or Response
        Target regeneration: only the resource status update itself
        (delegated entirely to OperationalRefreshOrchestrator.refresh_resource,
        unchanged), then planning reevaluation.

        A resource is not scoped to a single FireEvent in the current
        domain model: FirefightingResource references only its FireStation,
        and FireStation is not FireEvent-scoped. OperationalContextService's
        own station-selection radius is also progressive (5km -> 20km ->
        50km) with a closest-station-overall fallback, so no fixed-radius
        relationship implemented here could reliably reproduce which
        FireEvents actually depend on a given resource without duplicating
        that logic. Instead, every currently active FireEvent is checked as
        a candidate; ResponsePlanningRefreshOrchestrator's own planning
        fingerprint - never reproduced here - cheaply resolves any FireEvent
        this resource did not actually affect to NO_OP, with no new
        RoutePlanningRun/ResponsePlan/PlanComparison created for it.
        """
        operational_result = self._run_operational_resource_refresh(resource_id, new_status)
        if not operational_result.success:
            return OperationalPlanningRefreshResult(operational_result=operational_result, planning_results=())

        active_fire_event_ids = self._fire_event_repository.get_active_fire_event_ids()
        planning_results = tuple(
            self._planning_refresh.refresh(fire_event_id=fire_event_id, as_of=as_of)
            for fire_event_id in active_fire_event_ids
        )
        return OperationalPlanningRefreshResult(
            operational_result=operational_result,
            planning_results=planning_results,
        )

    def _run_operational_fire_event_refresh(
        self,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> OperationalRefreshResult:
        try:
            return self._operational_refresh_orchestrator.refresh_fire_event(
                fire_event_id=fire_event_id,
                trigger_type=trigger_type,
                as_of=as_of,
            )
        except Exception as exc:  # noqa: BLE001 - conservative orchestration failure boundary.
            logger.exception("Operational refresh failed for FireEvent %s", fire_event_id)
            return OperationalRefreshResult(
                trigger_type=trigger_type,
                status=OperationalRefreshStatus.FAILED,
                success=False,
                fire_event_id=fire_event_id,
                as_of=as_of,
                error_message=str(exc) or "Operational refresh failed.",
            )

    def _run_operational_resource_refresh(
        self,
        resource_id: int | str,
        new_status: ResourceStatus,
    ) -> OperationalRefreshResult:
        try:
            return self._operational_refresh_orchestrator.refresh_resource(
                resource_id=resource_id,
                new_status=new_status,
            )
        except Exception as exc:  # noqa: BLE001 - conservative orchestration failure boundary.
            logger.exception("Operational refresh failed for resource %r", resource_id)
            return OperationalRefreshResult(
                trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
                status=OperationalRefreshStatus.FAILED,
                success=False,
                error_message=str(exc) or "Operational refresh failed.",
            )
