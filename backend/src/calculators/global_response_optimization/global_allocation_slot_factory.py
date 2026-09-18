"""Pure Stage-5 demand-aware allocation-slot generation (Tasks 4/7/8/9/10),
extended in Stage 6 (Task 7/section 8) with hard-dispatch-lock overflow.

For each active FireEvent's canonical ACTIVE_FIRE target, generates
`max(desired_resources, locked_resource_count)` suppression slots: the
first `minimum_resources` are `required=True`, the next
`desired_resources - minimum_resources` are `required=False` desired/
optional reinforcement (Task 8). If more resources are already
HARD-DISPATCHED to this event than `desired_resources` calls for (Stage 6,
"demand decrease must never fabricate an implicit recall" - a configured
demand DECREASE must never silently evict an already-dispatched resource),
additional slots beyond `desired_resources` are generated - still
`required=False` - purely to give each already-dispatched resource a legal
structural home. These are LOCKED-OVERFLOW slots: they are never counted
toward severity-driven required/desired demand accounting (see
GlobalEventResourceDemandResult.locked_resources_preserved, which reports
them separately) - only toward how many slots physically exist for the
chromosome to assign locked resources onto.

Every PREDICTED_RISK target gets exactly one optional (`required=False`)
coverage slot, unconditionally - severity demand controls SUPPRESSION count
only; predicted-risk coverage is additional operational coverage, never
counted toward minimum_resources (Task 9).

Still no DB reads, no severity calculation - `incident_demands` and
`locked_resource_counts` are already built and handed in.
"""
from __future__ import annotations

from src.models.global_allocation_slot import GlobalAllocationSlot
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_target import GlobalPlanningTarget
from src.models.response_target_type import ResponseTargetType


class GlobalAllocationSlotGenerationError(ValueError):
    """Raised when demand and targets are inconsistent (Task 10): an event
    needs at least one suppression resource but has no ACTIVE_FIRE target
    to suppress. Never fabricates a target or coordinates to work around it."""


class GlobalAllocationSlotFactory:
    """Builds demand-aware GlobalAllocationSlots: several per ACTIVE_FIRE target, one per PREDICTED_RISK target."""

    @staticmethod
    def generate(
        targets: tuple[GlobalPlanningTarget, ...],
        incident_demands: tuple[GlobalIncidentDemand, ...],
        locked_resource_counts: dict[int, int] | None = None,
    ) -> tuple[GlobalAllocationSlot, ...]:
        target_tuple = _coerce_targets(targets)
        demand_tuple = _coerce_demands(incident_demands)
        demand_by_event = {demand.fire_event_id: demand for demand in demand_tuple}
        locked_counts = _coerce_locked_counts(locked_resource_counts)

        targets_by_event: dict[int, list[GlobalPlanningTarget]] = {}
        for target in target_tuple:
            targets_by_event.setdefault(target.fire_event_id, []).append(target)

        slots: list[GlobalAllocationSlot] = []
        event_ids = sorted(set(targets_by_event) | set(locked_counts))
        for fire_event_id in event_ids:
            event_targets = targets_by_event.get(fire_event_id, [])
            active_fire_targets = sorted(
                (t for t in event_targets if t.target_type is ResponseTargetType.ACTIVE_FIRE),
                key=lambda t: (t.target_order, t.response_target_id),
            )
            predicted_targets = sorted(
                (t for t in event_targets if t.target_type is ResponseTargetType.PREDICTED_RISK),
                key=lambda t: (t.target_order, t.response_target_id),
            )
            demand = demand_by_event.get(fire_event_id)
            desired_resources = demand.desired_resources if demand is not None else 0
            minimum_resources = demand.minimum_resources if demand is not None else 0
            locked_count = locked_counts.get(fire_event_id, 0)
            effective_slot_count = max(desired_resources, locked_count)

            if effective_slot_count > 0:
                if not active_fire_targets:
                    raise GlobalAllocationSlotGenerationError(
                        f"FireEvent {fire_event_id!r} has demand (minimum={minimum_resources!r}, "
                        f"desired={desired_resources!r}) or {locked_count!r} hard-dispatched resource(s) "
                        "but no ACTIVE_FIRE target to suppress."
                    )
                canonical_active_fire_target = active_fire_targets[0]
                for slot_index in range(effective_slot_count):
                    slots.append(
                        GlobalAllocationSlot(
                            slot_id=f"{canonical_active_fire_target.response_target_id}#{slot_index}",
                            fire_event_id=fire_event_id,
                            response_target_id=canonical_active_fire_target.response_target_id,
                            slot_index=slot_index,
                            required=slot_index < minimum_resources,
                        )
                    )

            for predicted_target in predicted_targets:
                slots.append(
                    GlobalAllocationSlot(
                        slot_id=f"{predicted_target.response_target_id}#0",
                        fire_event_id=fire_event_id,
                        response_target_id=predicted_target.response_target_id,
                        slot_index=0,
                        required=False,
                    )
                )

        return tuple(
            sorted(slots, key=lambda slot: (slot.fire_event_id, slot.response_target_id, slot.slot_index))
        )


def _coerce_targets(targets: object) -> tuple[GlobalPlanningTarget, ...]:
    try:
        target_tuple = tuple(targets)
    except TypeError as exc:
        raise ValueError("targets must be iterable.") from exc
    for target in target_tuple:
        if not isinstance(target, GlobalPlanningTarget):
            raise ValueError(f"targets must contain GlobalPlanningTarget items, got {target!r}")
    return target_tuple


def _coerce_locked_counts(locked_resource_counts: object) -> dict[int, int]:
    if locked_resource_counts is None:
        return {}
    if not isinstance(locked_resource_counts, dict):
        raise ValueError(f"locked_resource_counts must be a dict or None, got {locked_resource_counts!r}")
    for fire_event_id, count in locked_resource_counts.items():
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"locked_resource_counts keys must be positive integers, got {fire_event_id!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"locked_resource_counts values must be non-negative integers, got {count!r}")
    return dict(locked_resource_counts)


def _coerce_demands(incident_demands: object) -> tuple[GlobalIncidentDemand, ...]:
    try:
        demand_tuple = tuple(incident_demands)
    except TypeError as exc:
        raise ValueError("incident_demands must be iterable.") from exc
    for demand in demand_tuple:
        if not isinstance(demand, GlobalIncidentDemand):
            raise ValueError(f"incident_demands must contain GlobalIncidentDemand items, got {demand!r}")
    fire_event_ids = [demand.fire_event_id for demand in demand_tuple]
    if len(set(fire_event_ids)) != len(fire_event_ids):
        raise ValueError("incident_demands must not contain duplicate fire_event_id entries.")
    return demand_tuple
