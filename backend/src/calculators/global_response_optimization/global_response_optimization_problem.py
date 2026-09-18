"""GlobalResponseOptimizationProblem: precomputed, GA-ready view of one
GlobalPlanningInput (Stage 4, Task 8; hard-dispatch-lock restriction and
soft-stability lookup added Stage 6).

Built ONCE per optimize() call; the GA never rebuilds these maps during
fitness evaluation. Purely derived from the supplied GlobalPlanningInput -
no repository/DB/Dijkstra access, matching Task 1's architectural boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.calculators.global_response_optimization.global_allocation_slot_factory import GlobalAllocationSlotFactory
from src.calculators.global_response_optimization.global_hard_dispatch_lock import GlobalHardDispatchLockInfeasible
from src.models.current_global_assignment import CurrentGlobalAssignment
from src.models.dispatch_state import DispatchState
from src.models.global_allocation_slot import GlobalAllocationSlot
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_planning_resource import GlobalPlanningResource
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.global_route_matrix import GlobalRouteMatrix
from src.models.global_route_option import GlobalRouteOption


@dataclass(frozen=True)
class GlobalResponseOptimizationProblem:
    """Everything the Global GA needs, indexed for O(1) lookup during evolution."""

    global_planning_run_id: int
    input_fingerprint: str
    resource_ids: tuple[str, ...]
    resources_by_id: dict[str, GlobalPlanningResource]
    slots: tuple[GlobalAllocationSlot, ...]
    slots_by_id: dict[str, GlobalAllocationSlot]
    targets_by_id: dict[int, GlobalPlanningTarget]
    route_matrix: GlobalRouteMatrix
    incident_demands_by_event: dict[int, GlobalIncidentDemand]
    current_assignment_by_resource: dict[str, CurrentGlobalAssignment]
    locked_resource_ids: frozenset[str]
    feasible_slot_ids_by_resource: dict[str, tuple[str, ...]] = field(repr=False)

    def route_for(self, resource_id: str, slot_id: str) -> GlobalRouteOption | None:
        """The feasible route for a resource-slot pair, or None if infeasible/absent."""
        slot = self.slots_by_id.get(slot_id)
        if slot is None:
            return None
        return self.route_matrix.get(resource_id, slot.response_target_id)


def build_global_optimization_problem(global_planning_input: GlobalPlanningInput) -> GlobalResponseOptimizationProblem:
    """Build one GlobalResponseOptimizationProblem from a GlobalPlanningInput.

    Only ASSIGNABLE resources (Task 7: everything except UNAVAILABLE) enter
    resource_ids - the GA's gene universe. Deterministic resource ordering
    (ascending resource_id) so chromosome gene positions are stable across
    calls for the same input.

    Stage 6: every resource whose current_assignment.dispatch_state is
    DISPATCHED has its feasible_slot_ids_by_resource entry restricted to
    ONLY slots belonging to its locked fire_event_id - this is what makes
    "cross to another FireEvent" structurally unreachable for the GA (see
    global_hard_dispatch_lock.py), not merely penalized. The slot factory
    is also told each event's locked-resource count, so enough slots exist
    to legally host every already-dispatched resource even when severity-
    driven desired demand alone would call for fewer (a demand decrease
    must never fabricate an implicit recall).
    """
    if not isinstance(global_planning_input, GlobalPlanningInput):
        raise ValueError(f"global_planning_input must be a GlobalPlanningInput, got {global_planning_input!r}")

    assignable_resources = tuple(
        sorted(
            (resource for resource in global_planning_input.resources if resource.is_assignable),
            key=lambda resource: resource.resource_id,
        )
    )
    resource_ids = tuple(resource.resource_id for resource in assignable_resources)
    resources_by_id = {resource.resource_id: resource for resource in assignable_resources}

    current_assignment_by_resource = {
        assignment.resource_id: assignment
        for assignment in global_planning_input.current_assignments
        if assignment.resource_id in resources_by_id
    }
    locked_resource_ids = frozenset(
        resource_id
        for resource_id, assignment in current_assignment_by_resource.items()
        if assignment.dispatch_state is DispatchState.DISPATCHED
    )
    locked_resource_counts: dict[int, int] = {}
    for resource_id in locked_resource_ids:
        fire_event_id = current_assignment_by_resource[resource_id].fire_event_id
        locked_resource_counts[fire_event_id] = locked_resource_counts.get(fire_event_id, 0) + 1

    slots = GlobalAllocationSlotFactory.generate(
        global_planning_input.targets, global_planning_input.incident_demands, locked_resource_counts
    )
    slots_by_id = {slot.slot_id: slot for slot in slots}
    targets_by_id = {target.response_target_id: target for target in global_planning_input.targets}
    incident_demands_by_event = {demand.fire_event_id: demand for demand in global_planning_input.incident_demands}

    feasible_slot_ids_by_resource: dict[str, list[str]] = {resource_id: [] for resource_id in resource_ids}
    for slot in slots:
        for resource_id in resource_ids:
            if global_planning_input.route_matrix.get(resource_id, slot.response_target_id) is not None:
                feasible_slot_ids_by_resource[resource_id].append(slot.slot_id)

    for resource_id in locked_resource_ids:
        locked_fire_event_id = current_assignment_by_resource[resource_id].fire_event_id
        feasible_slot_ids_by_resource[resource_id] = [
            slot_id
            for slot_id in feasible_slot_ids_by_resource[resource_id]
            if slots_by_id[slot_id].fire_event_id == locked_fire_event_id
        ]
        if not feasible_slot_ids_by_resource[resource_id]:
            raise GlobalHardDispatchLockInfeasible(
                f"hard-dispatched resource {resource_id!r} has no feasible slot for its locked "
                f"FireEvent {locked_fire_event_id!r} - cannot represent its required hard lock."
            )

    return GlobalResponseOptimizationProblem(
        global_planning_run_id=global_planning_input.global_planning_run_id,
        input_fingerprint=global_planning_input.input_fingerprint,
        resource_ids=resource_ids,
        resources_by_id=resources_by_id,
        slots=slots,
        slots_by_id=slots_by_id,
        targets_by_id=targets_by_id,
        route_matrix=global_planning_input.route_matrix,
        incident_demands_by_event=incident_demands_by_event,
        current_assignment_by_resource=current_assignment_by_resource,
        locked_resource_ids=locked_resource_ids,
        feasible_slot_ids_by_resource={
            resource_id: tuple(slot_ids) for resource_id, slot_ids in feasible_slot_ids_by_resource.items()
        },
    )
