"""US 4.4 -> Stage 6 global planning orchestration bridge.

OperationalPlanningRefreshCoordinator is a thin composition layer: it calls
OperationalRefreshOrchestrator (US 4.4), and - only if that refresh
succeeded - calls a GlobalPlanningRefreshPort (GlobalPlanningRefreshCoordinator
in production, Stage 6) exactly ONCE, covering every currently active
FireEvent together. It contains no severity/spread/target logic of its
own, and no Dijkstra/GA/baseline logic of its own: whether the global cycle
actually reruns the GA, or is a NO_OP, is entirely
GlobalPlanningRefreshCoordinator's own input+policy fingerprint decision -
not decided or precomputed here.

Stage 6 Task 47 (production cutover): this coordinator used to call a
PlanningRefreshPort once per affected FireEvent - refresh(A); refresh(B);
refresh(C) through independent legacy per-event planners. It now calls the
global port once per operational change, regardless of how many FireEvents
are active - "one material change -> one global refresh -> one atomic
activation," never a per-event fan-out. The legacy per-event port/
orchestrator remain in source (PlanningRefreshPort, ResponsePlanningRefreshOrchestrator)
for direct/historical use; they are simply no longer wired here.
"""
from __future__ import annotations

from datetime import datetime
import logging

from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.models.resource_status import ResourceStatus
from src.services.operational_planning_refresh.operational_planning_refresh_ports import GlobalPlanningRefreshPort
from src.services.operational_planning_refresh.operational_planning_refresh_result import (
    OperationalPlanningRefreshBatchResult,
    OperationalPlanningRefreshResult,
)
from src.services.operational_refresh.operational_refresh_orchestrator import OperationalRefreshOrchestrator
from src.services.operational_refresh.operational_refresh_result import (
    OperationalRefreshResult,
    OperationalRefreshStatus,
)

logger = logging.getLogger(__name__)


class OperationalPlanningRefreshCoordinator:
    """Trigger ONE Stage 6 global planning refresh after a US 4.4 operational refresh."""

    def __init__(
        self,
        *,
        operational_refresh_orchestrator: OperationalRefreshOrchestrator,
        global_planning_refresh: GlobalPlanningRefreshPort,
    ) -> None:
        self._operational_refresh_orchestrator = operational_refresh_orchestrator
        self._global_planning_refresh = global_planning_refresh

    def refresh_fire_event(
        self,
        *,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> OperationalPlanningRefreshResult:
        """Run one US 4.4 environmental/FireEvent refresh, then ONE global planning refresh.

        The global refresh covers every currently active FireEvent, not
        just `fire_event_id` - there is one global optimization problem
        (Task 1). It only runs after severity/spread/response-target
        refresh has completed (severity, then spread, then response
        targets, in that existing order), so it observes the newest
        persisted planning inputs, and never runs after an operational
        failure.
        """
        operational_result = self._run_operational_fire_event_refresh(fire_event_id, trigger_type, as_of)
        if not operational_result.success:
            return OperationalPlanningRefreshResult(operational_result=operational_result, global_planning_result=None)

        global_planning_result = self._global_planning_refresh.refresh(trigger=trigger_type.value, as_of=as_of)
        return OperationalPlanningRefreshResult(
            operational_result=operational_result,
            global_planning_result=global_planning_result,
        )

    def refresh_fire_events_batch(
        self,
        *,
        fire_event_ids: tuple[int, ...],
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> OperationalPlanningRefreshBatchResult:
        """Run one US 4.4 operational refresh per FireEvent in `fire_event_ids`,
        then ONE global planning refresh for the whole batch - never one
        global refresh per FireEvent (Task 34/35). Use this instead of
        calling `refresh_fire_event` in a loop whenever several FireEvents
        share one logical upstream update (e.g. a weather update affecting
        several nearby active fires at once).
        """
        operational_results = tuple(
            self._run_operational_fire_event_refresh(fire_event_id, trigger_type, as_of)
            for fire_event_id in fire_event_ids
        )
        if not any(result.success for result in operational_results):
            return OperationalPlanningRefreshBatchResult(
                operational_results=operational_results, global_planning_result=None
            )

        global_planning_result = self._global_planning_refresh.refresh(trigger=trigger_type.value, as_of=as_of)
        return OperationalPlanningRefreshBatchResult(
            operational_results=operational_results,
            global_planning_result=global_planning_result,
        )

    def refresh_resource(
        self,
        *,
        resource_id: int | str,
        new_status: ResourceStatus,
        as_of: datetime,
    ) -> OperationalPlanningRefreshResult:
        """Run one US 4.4 resource-status update, then ONE global planning refresh.

        Runs no Fire Detection, Fire Severity, Fire Spread, or Response
        Target regeneration: only the resource status update itself
        (delegated entirely to OperationalRefreshOrchestrator.refresh_resource,
        unchanged), then one global planning reevaluation covering every
        currently active FireEvent together - never a per-FireEvent fan-out
        (Stage 6, Task 35: "R1 AVAILABLE -> UNAVAILABLE should trigger ONE
        GlobalPlanningRefreshCoordinator.refresh(...) for the global active
        set").
        """
        operational_result = self._run_operational_resource_refresh(resource_id, new_status)
        if not operational_result.success:
            return OperationalPlanningRefreshResult(operational_result=operational_result, global_planning_result=None)

        global_planning_result = self._global_planning_refresh.refresh(
            trigger=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE.value, as_of=as_of
        )
        return OperationalPlanningRefreshResult(
            operational_result=operational_result,
            global_planning_result=global_planning_result,
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
