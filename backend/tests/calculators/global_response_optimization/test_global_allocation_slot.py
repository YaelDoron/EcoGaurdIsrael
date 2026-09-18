"""Tests for GlobalAllocationSlot and GlobalAllocationSlotFactory (Stage 4
Tasks 3-4; demand-aware multi-slot generation since Stage 5, Tasks 7-10)."""
from __future__ import annotations

import pytest

from src.calculators.global_response_optimization.global_allocation_slot_factory import (
    GlobalAllocationSlotFactory,
    GlobalAllocationSlotGenerationError,
)
from src.models.global_allocation_slot import GlobalAllocationSlot
from src.models.response_target_type import ResponseTargetType
from tests.calculators.global_response_optimization.helpers import make_demand, make_target


def test_valid_slot():
    slot = GlobalAllocationSlot(slot_id="10#0", fire_event_id=1, response_target_id=10, slot_index=0, required=False)
    assert slot.slot_index == 0
    assert slot.required is False


def test_rejects_invalid_ids():
    with pytest.raises(ValueError):
        GlobalAllocationSlot(slot_id="", fire_event_id=1, response_target_id=10, slot_index=0, required=False)
    with pytest.raises(ValueError):
        GlobalAllocationSlot(slot_id="s", fire_event_id=0, response_target_id=10, slot_index=0, required=False)
    with pytest.raises(ValueError):
        GlobalAllocationSlot(slot_id="s", fire_event_id=1, response_target_id=0, slot_index=0, required=False)


def test_rejects_negative_slot_index():
    with pytest.raises(ValueError):
        GlobalAllocationSlot(slot_id="s", fire_event_id=1, response_target_id=10, slot_index=-1, required=False)


def test_rejects_non_bool_required():
    with pytest.raises(ValueError):
        GlobalAllocationSlot(slot_id="s", fire_event_id=1, response_target_id=10, slot_index=0, required="no")


# ---------------------------------------------------------------------------
# GlobalAllocationSlotFactory - neutral demand (Stage 4 parity)
# ---------------------------------------------------------------------------


def test_neutral_demand_creates_exactly_one_optional_slot_per_target():
    """Stage 4 parity: minimum=0/desired=1 reduces to one optional slot per target."""
    targets = (
        make_target(1, 10, target_order=1, target_type=ResponseTargetType.ACTIVE_FIRE),
        make_target(1, 11, target_order=2, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30),
        make_target(2, 20, target_order=1, target_type=ResponseTargetType.ACTIVE_FIRE),
    )
    demands = (make_demand(1), make_demand(2))

    slots = GlobalAllocationSlotFactory.generate(targets, demands)

    assert len(slots) == 3
    assert all(slot.slot_index == 0 for slot in slots)
    assert all(slot.required is False for slot in slots)
    assert {slot.response_target_id for slot in slots} == {10, 11, 20}


def test_slot_ids_are_unique_and_deterministic():
    targets = (make_target(1, 10), make_target(1, 11), make_target(2, 20))
    demands = (make_demand(1), make_demand(2))

    first = GlobalAllocationSlotFactory.generate(targets, demands)
    second = GlobalAllocationSlotFactory.generate(targets, demands)

    slot_ids = [slot.slot_id for slot in first]
    assert len(slot_ids) == len(set(slot_ids))
    assert first == second


def test_factory_rejects_non_target_items():
    with pytest.raises(ValueError):
        GlobalAllocationSlotFactory.generate(("not-a-target",), ())


def test_factory_empty_targets_and_demands_produces_empty_slots():
    assert GlobalAllocationSlotFactory.generate((), ()) == ()


# ---------------------------------------------------------------------------
# Task 7/8 - multi-slot suppression demand, required vs desired
# ---------------------------------------------------------------------------


def test_critical_demand_produces_several_distinct_required_and_desired_slots():
    active_fire_target = make_target(1, 10, target_type=ResponseTargetType.ACTIVE_FIRE)
    demand = make_demand(1, minimum_resources=3, desired_resources=4)

    slots = GlobalAllocationSlotFactory.generate((active_fire_target,), (demand,))

    assert len(slots) == 4
    assert all(slot.response_target_id == 10 for slot in slots)  # same physical fire target
    slot_ids = [slot.slot_id for slot in slots]
    assert len(slot_ids) == len(set(slot_ids))  # distinct slot ids
    required_flags = sorted(slot.required for slot in slots)
    assert required_flags == [False, True, True, True]  # 3 required, 1 desired


def test_required_slots_are_exactly_the_first_minimum_slot_indices():
    active_fire_target = make_target(1, 10)
    demand = make_demand(1, minimum_resources=2, desired_resources=3)

    slots = GlobalAllocationSlotFactory.generate((active_fire_target,), (demand,))

    by_index = {slot.slot_index: slot for slot in slots}
    assert by_index[0].required is True
    assert by_index[1].required is True
    assert by_index[2].required is False


# ---------------------------------------------------------------------------
# Task 9 - predicted-risk targets always get exactly one optional slot
# ---------------------------------------------------------------------------


def test_predicted_risk_target_gets_one_optional_slot_regardless_of_demand():
    active_fire_target = make_target(1, 10, target_type=ResponseTargetType.ACTIVE_FIRE)
    predicted_target = make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30)
    demand = make_demand(1, minimum_resources=3, desired_resources=3)  # high suppression demand

    slots = GlobalAllocationSlotFactory.generate((active_fire_target, predicted_target), (demand,))

    predicted_slots = [slot for slot in slots if slot.response_target_id == 11]
    assert len(predicted_slots) == 1
    assert predicted_slots[0].required is False


def test_predicted_risk_slots_never_increase_minimum_resources():
    """Task 9: predicted-risk coverage is additional optional coverage, never
    counted toward minimum_resources - severity controls suppression demand only."""
    active_fire_target = make_target(1, 10)
    predicted_targets = tuple(
        make_target(1, 20 + i, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30)
        for i in range(3)
    )
    demand = make_demand(1, minimum_resources=1, desired_resources=1)

    slots = GlobalAllocationSlotFactory.generate((active_fire_target, *predicted_targets), (demand,))

    required_slots = [slot for slot in slots if slot.required]
    assert len(required_slots) == 1  # still just 1, unaffected by the 3 predicted-risk targets
    assert len(slots) == 1 + 3  # 1 suppression slot + 3 predicted-risk slots


# ---------------------------------------------------------------------------
# Task 10 - missing ACTIVE_FIRE target with positive demand
# ---------------------------------------------------------------------------


def test_missing_active_fire_target_with_positive_demand_raises():
    predicted_target = make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30)
    demand = make_demand(1, minimum_resources=1, desired_resources=2)

    with pytest.raises(GlobalAllocationSlotGenerationError):
        GlobalAllocationSlotFactory.generate((predicted_target,), (demand,))


def test_missing_active_fire_target_with_zero_demand_does_not_raise():
    predicted_target = make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30)
    demand = make_demand(1, minimum_resources=0, desired_resources=0)

    slots = GlobalAllocationSlotFactory.generate((predicted_target,), (demand,))

    assert len(slots) == 1
    assert slots[0].response_target_id == 11


def test_factory_rejects_duplicate_demand_fire_event_ids():
    target = make_target(1, 10)
    with pytest.raises(ValueError):
        GlobalAllocationSlotFactory.generate((target,), (make_demand(1), make_demand(1)))


# ---------------------------------------------------------------------------
# Stage 6 - locked-overflow slots (hard-dispatched resources beyond desired)
# ---------------------------------------------------------------------------


def test_locked_resource_count_below_desired_adds_no_overflow_slots():
    target = make_target(1, 10)
    demand = make_demand(1, minimum_resources=1, desired_resources=3)

    slots = GlobalAllocationSlotFactory.generate((target,), (demand,), locked_resource_counts={1: 2})

    assert len(slots) == 3


def test_locked_resource_count_above_desired_adds_overflow_slots():
    target = make_target(1, 10)
    demand = make_demand(1, minimum_resources=1, desired_resources=2)

    slots = GlobalAllocationSlotFactory.generate((target,), (demand,), locked_resource_counts={1: 4})

    assert len(slots) == 4
    assert sum(1 for slot in slots if slot.required) == 1
    assert sum(1 for slot in slots if not slot.required) == 3


def test_locked_overflow_slots_never_marked_required():
    target = make_target(1, 10)
    demand = make_demand(1, minimum_resources=0, desired_resources=0)

    slots = GlobalAllocationSlotFactory.generate((target,), (demand,), locked_resource_counts={1: 2})

    assert len(slots) == 2
    assert all(not slot.required for slot in slots)


def test_locked_resource_count_with_no_demand_entry_still_generates_slots():
    target = make_target(1, 10)

    slots = GlobalAllocationSlotFactory.generate((target,), (), locked_resource_counts={1: 2})

    assert len(slots) == 2
    assert all(not slot.required for slot in slots)


def test_locked_resource_count_with_no_active_fire_target_raises():
    predicted_target = make_target(1, 11, target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30)

    with pytest.raises(GlobalAllocationSlotGenerationError):
        GlobalAllocationSlotFactory.generate((predicted_target,), (), locked_resource_counts={1: 1})


def test_locked_resource_counts_rejects_invalid_types():
    target = make_target(1, 10)
    with pytest.raises(ValueError):
        GlobalAllocationSlotFactory.generate((target,), (), locked_resource_counts={1: -1})
    with pytest.raises(ValueError):
        GlobalAllocationSlotFactory.generate((target,), (), locked_resource_counts="not-a-dict")
